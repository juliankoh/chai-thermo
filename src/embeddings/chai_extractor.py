"""Extract trunk representations (single + pair) from Chai-1.

This module runs Chai-1 up through the trunk but skips diffusion,
giving us the learned representations for stability prediction.

Requires: Linux + CUDA GPU (A100/H100 recommended)
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from torch import Tensor
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ChaiEmbeddings:
    """Trunk embeddings for a single protein."""

    protein_name: str
    sequence: str
    single: Tensor  # [L, D_single] - per-residue features
    pair: Tensor  # [L, L, D_pair] - pairwise interaction features

    def save(self, path: Path) -> None:
        """Save embeddings to disk."""
        torch.save(
            {
                "protein_name": self.protein_name,
                "sequence": self.sequence,
                "single": self.single,
                "pair": self.pair,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "ChaiEmbeddings":
        """Load embeddings from disk."""
        data = torch.load(path, map_location="cpu", weights_only=False)
        return cls(**data)


def extract_trunk_embeddings(
    sequence: str,
    protein_name: str = "protein",
    device: str = "cuda:0",
    num_trunk_recycles: int = 3,
    use_esm_embeddings: bool = True,
) -> ChaiEmbeddings:
    """
    Extract single and pair representations from Chai-1 trunk.

    This runs the Chai-1 pipeline up through the trunk module,
    skipping diffusion (structure prediction) to get embeddings only.

    Args:
        sequence: Amino acid sequence (single letter codes)
        protein_name: Identifier for the protein
        device: CUDA device to use
        num_trunk_recycles: Number of trunk recycle iterations
        use_esm_embeddings: Whether to use ESM embeddings (recommended)

    Returns:
        ChaiEmbeddings with single [L, D_single] and pair [L, L, D_pair] tensors
    """
    import tempfile
    import shutil

    # Import chai-lab components
    from chai_lab.chai1 import (
        make_all_atom_feature_context,
        feature_factory,
        _component_moved_to,
        raise_if_too_many_tokens,
    )
    from chai_lab.data.collate.collate import Collate
    from chai_lab.data.collate.utils import AVAILABLE_MODEL_SIZES
    from chai_lab.data.features.generators.token_bond import TokenBondRestraint
    from chai_lab.utils.tensor_utils import move_data_to_device, und_self

    torch_device = torch.device(device)

    # Create temporary FASTA file
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        fasta_path = tmpdir / "input.fasta"
        output_dir = tmpdir / "output"
        output_dir.mkdir()

        # Write FASTA
        fasta_content = f">protein|name={protein_name}\n{sequence}\n"
        fasta_path.write_text(fasta_content)

        # Build feature context (no MSA, no templates for speed)
        logger.info(f"Building feature context for {protein_name} ({len(sequence)} aa)")
        feature_context = make_all_atom_feature_context(
            fasta_file=fasta_path,
            output_dir=output_dir,
            use_esm_embeddings=use_esm_embeddings,
            use_msa_server=False,
            msa_directory=None,
            constraint_path=None,
            use_templates_server=False,
            templates_path=None,
            esm_device=torch_device,
        )

    n_tokens = feature_context.structure_context.num_tokens
    raise_if_too_many_tokens(n_tokens)

    # Collate into batch
    collator = Collate(
        feature_factory=feature_factory,
        num_key_atoms=128,
        num_query_atoms=32,
    )
    batch = collator([feature_context])

    # Get features and inputs
    features = {name: feat for name, feat in batch["features"].items()}
    inputs = batch["inputs"]

    block_indices_h = inputs["block_atom_pair_q_idces"]
    block_indices_w = inputs["block_atom_pair_kv_idces"]
    atom_single_mask = inputs["atom_exists_mask"]
    atom_token_indices = inputs["atom_token_index"].long()
    token_single_mask = inputs["token_exists_mask"]
    token_pair_mask = und_self(token_single_mask, "b i, b j -> b i j")
    msa_mask = inputs["msa_mask"]
    template_input_masks = und_self(
        inputs["template_mask"], "b t n1, b t n2 -> b t n1 n2"
    )
    block_atom_pair_mask = inputs["block_atom_pair_mask"]

    # Determine model size
    _, _, model_size = msa_mask.shape
    assert model_size in AVAILABLE_MODEL_SIZES

    logger.info(f"Running feature embedding (model size: {model_size})")

    # === Feature Embedding ===
    with _component_moved_to("feature_embedding.pt", torch_device) as feature_embedding:
        embedded_features = feature_embedding.forward(
            crop_size=model_size,
            move_to_device=torch_device,
            return_on_cpu=False,
            **features,
        )

    token_single_input_feats = embedded_features["TOKEN"]
    token_pair_input_feats, token_pair_structure_input_feats = embedded_features[
        "TOKEN_PAIR"
    ].chunk(2, dim=-1)
    atom_single_input_feats, atom_single_structure_input_feats = embedded_features[
        "ATOM"
    ].chunk(2, dim=-1)
    block_atom_pair_input_feats, block_atom_pair_structure_input_feats = (
        embedded_features["ATOM_PAIR"].chunk(2, dim=-1)
    )
    template_input_feats = embedded_features["TEMPLATES"]
    msa_input_feats = embedded_features["MSA"]

    # === Bond Features ===
    bond_ft_gen = TokenBondRestraint()
    bond_ft = bond_ft_gen.generate(batch=batch).data
    with _component_moved_to("bond_loss_input_proj.pt", torch_device) as bond_proj:
        trunk_bond_feat, structure_bond_feat = bond_proj.forward(
            return_on_cpu=False,
            move_to_device=torch_device,
            crop_size=model_size,
            input=bond_ft,
        ).chunk(2, dim=-1)
    token_pair_input_feats = token_pair_input_feats + trunk_bond_feat

    # === Token Embedder ===
    logger.info("Running token embedder")
    with _component_moved_to("token_embedder.pt", torch_device) as token_embedder:
        token_embedder_outputs = token_embedder.forward(
            return_on_cpu=False,
            move_to_device=torch_device,
            token_single_input_feats=token_single_input_feats,
            token_pair_input_feats=token_pair_input_feats,
            atom_single_input_feats=atom_single_input_feats,
            block_atom_pair_feat=block_atom_pair_input_feats,
            block_atom_pair_mask=block_atom_pair_mask,
            block_indices_h=block_indices_h,
            block_indices_w=block_indices_w,
            atom_single_mask=atom_single_mask,
            atom_token_indices=atom_token_indices,
            crop_size=model_size,
        )

    token_single_initial_repr, token_single_structure_input, token_pair_initial_repr = (
        token_embedder_outputs
    )

    # === Trunk (with recycles) ===
    logger.info(f"Running trunk ({num_trunk_recycles} recycles)")
    token_single_trunk_repr = token_single_initial_repr
    token_pair_trunk_repr = token_pair_initial_repr

    for recycle_idx in range(num_trunk_recycles):
        with _component_moved_to("trunk.pt", torch_device) as trunk:
            token_single_trunk_repr, token_pair_trunk_repr = trunk.forward(
                move_to_device=torch_device,
                token_single_trunk_initial_repr=token_single_initial_repr,
                token_pair_trunk_initial_repr=token_pair_initial_repr,
                token_single_trunk_repr=token_single_trunk_repr,
                token_pair_trunk_repr=token_pair_trunk_repr,
                msa_input_feats=msa_input_feats,
                msa_mask=msa_mask,
                template_input_feats=template_input_feats,
                template_input_masks=template_input_masks,
                token_single_mask=token_single_mask,
                token_pair_mask=token_pair_mask,
                crop_size=model_size,
            )

    # Extract embeddings (remove batch dimension, move to CPU)
    # token_single_trunk_repr: [1, L, D_single]
    # token_pair_trunk_repr: [1, L, L, D_pair]
    single_emb = token_single_trunk_repr[0].cpu()  # [L, D_single]
    pair_emb = token_pair_trunk_repr[0].cpu()  # [L, L, D_pair]

    logger.info(f"Extracted embeddings: single={single_emb.shape}, pair={pair_emb.shape}")

    return ChaiEmbeddings(
        protein_name=protein_name,
        sequence=sequence,
        single=single_emb,
        pair=pair_emb,
    )


def extract_all_proteins(
    sequences: dict[str, str],
    output_dir: Path,
    device: str = "cuda:0",
    num_trunk_recycles: int = 3,
    use_esm_embeddings: bool = True,
    skip_existing: bool = True,
) -> None:
    """
    Extract embeddings for all proteins and save to disk.

    Args:
        sequences: Dict of {protein_name: sequence}
        output_dir: Directory to save embeddings
        device: CUDA device
        num_trunk_recycles: Number of trunk recycles
        use_esm_embeddings: Whether to use ESM embeddings
        skip_existing: Skip proteins that already have embeddings
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, seq in tqdm(sequences.items(), desc="Extracting embeddings"):
        out_path = output_dir / f"{name}.pt"

        if skip_existing and out_path.exists():
            logger.info(f"Skipping {name} (already exists)")
            continue

        try:
            embeddings = extract_trunk_embeddings(
                sequence=seq,
                protein_name=name,
                device=device,
                num_trunk_recycles=num_trunk_recycles,
                use_esm_embeddings=use_esm_embeddings,
            )
            embeddings.save(out_path)
            logger.info(f"Saved {name} to {out_path}")

            # Clear GPU memory between proteins
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Failed to extract {name}: {e}")
            continue


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract Chai-1 trunk embeddings")
    parser.add_argument(
        "--sequences",
        type=Path,
        default=Path("data/wt_sequences.json"),
        help="JSON file with {name: sequence} dict",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/embeddings/chai_trunk"),
        help="Output directory for embeddings",
    )
    parser.add_argument("--device", default="cuda:0", help="CUDA device")
    parser.add_argument("--recycles", type=int, default=3, help="Number of trunk recycles")
    parser.add_argument("--no-esm", action="store_true", help="Disable ESM embeddings")
    parser.add_argument("--no-skip", action="store_true", help="Re-extract existing")

    args = parser.parse_args()

    # Load sequences
    with open(args.sequences) as f:
        sequences = json.load(f)

    print(f"Loaded {len(sequences)} sequences")
    print(f"Output directory: {args.output_dir}")

    extract_all_proteins(
        sequences=sequences,
        output_dir=args.output_dir,
        device=args.device,
        num_trunk_recycles=args.recycles,
        use_esm_embeddings=not args.no_esm,
        skip_existing=not args.no_skip,
    )
