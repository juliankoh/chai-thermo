"""Extract trunk representations (single + pair) from Chai-1.

This module runs Chai-1 up through the trunk but skips diffusion,
giving us the learned representations for stability prediction.

Optimized for A100/H100 GPUs - keeps models in VRAM for fast batch processing.

Requires: Linux + CUDA GPU (A100/H100 recommended)
"""

import json
import logging
import argparse
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from tqdm import tqdm

# chai-lab imports
from chai_lab.chai1 import load_exported, make_all_atom_feature_context, feature_factory
from chai_lab.data.collate.collate import Collate
from chai_lab.data.features.generators.token_bond import TokenBondRestraint
from chai_lab.utils.tensor_utils import und_self

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


class ChaiTrunkExtractor:
    """Extracts trunk embeddings from Chai-1.

    Keeps all model components loaded in VRAM for fast batch processing.
    Uses ~15-20GB VRAM total - on A100 80GB you can run multiple instances.
    """

    def __init__(self, device: str = "cuda:0"):
        self.device = torch.device(device)
        self.dtype = torch.bfloat16  # A100/H100 optimized

        logger.info(f"Loading Chai-1 to {self.device}...")

        # Load all components once, keep in VRAM
        self.feature_embedding = load_exported("feature_embedding.pt", self.device)
        self.bond_proj = load_exported("bond_loss_input_proj.pt", self.device)
        self.token_embedder = load_exported("token_embedder.pt", self.device)
        self.trunk = load_exported("trunk.pt", self.device)

        self.collator = Collate(
            feature_factory=feature_factory,
            num_key_atoms=128,
            num_query_atoms=32,
        )
        self.bond_ft_gen = TokenBondRestraint()

        logger.info("Chai-1 loaded successfully")

    @torch.inference_mode()
    def extract(
        self,
        sequence: str,
        protein_name: str = "protein",
        num_trunk_recycles: int = 3,
        use_esm_embeddings: bool = False,
    ) -> ChaiEmbeddings:
        """
        Extract single and pair representations from Chai-1 trunk.

        Args:
            sequence: Amino acid sequence (single letter codes)
            protein_name: Identifier for the protein
            num_trunk_recycles: Number of trunk recycle iterations
            use_esm_embeddings: Whether to use ESM embeddings (default: False, pure Chai)

        Returns:
            ChaiEmbeddings with single [L, D_single] and pair [L, L, D_pair] tensors
        """
        # --- 1. Data Prep ---
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            fasta_path = tmp_path / "input.fasta"
            fasta_path.write_text(f">protein|name={protein_name}\n{sequence}\n")

            logger.info(f"Building feature context for {protein_name} ({len(sequence)} aa)")
            feature_context = make_all_atom_feature_context(
                fasta_file=fasta_path,
                output_dir=tmp_path / "output",
                use_esm_embeddings=use_esm_embeddings,
                esm_device=self.device,  # Only used if use_esm_embeddings=True
                use_msa_server=False,
                use_templates_server=False,
            )

        # --- 2. Move to GPU ---
        # Note: batch contains mix of Tensors and Python lists (metadata)
        # Only move Tensors to device
        batch = self.collator([feature_context])
        features = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch["features"].items()
        }
        inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch["inputs"].items()
        }
        _, _, model_size = inputs["msa_mask"].shape

        # --- 3. Feature Embedding ---
        logger.info(f"Running feature embedding (model size: {model_size})")
        embedded = self.feature_embedding.forward(
            crop_size=model_size,
            move_to_device=self.device,
            return_on_cpu=False,
            **features,
        )

        # Cast to bfloat16
        tok_single = embedded["TOKEN"].to(self.dtype)
        tok_pair, _ = embedded["TOKEN_PAIR"].chunk(2, dim=-1)
        tok_pair = tok_pair.to(self.dtype)
        atom_single, _ = embedded["ATOM"].chunk(2, dim=-1)
        atom_single = atom_single.to(self.dtype)
        block_atom_pair, _ = embedded["ATOM_PAIR"].chunk(2, dim=-1)
        block_atom_pair = block_atom_pair.to(self.dtype)

        # --- 4. Bond Features ---
        bond_ft = self.bond_ft_gen.generate(batch=batch).data.to(self.device)
        trunk_bond_feat, _ = self.bond_proj.forward(
            crop_size=model_size,
            input=bond_ft,
            move_to_device=self.device,
            return_on_cpu=False,
        ).chunk(2, dim=-1)
        tok_pair = tok_pair + trunk_bond_feat.to(self.dtype)

        # --- 5. Token Embedder ---
        logger.info("Running token embedder")
        tok_single_curr, _, tok_pair_curr = self.token_embedder.forward(
            token_single_input_feats=tok_single,
            token_pair_input_feats=tok_pair,
            atom_single_input_feats=atom_single,
            block_atom_pair_feat=block_atom_pair,
            block_atom_pair_mask=inputs["block_atom_pair_mask"],
            block_indices_h=inputs["block_atom_pair_q_idces"],
            block_indices_w=inputs["block_atom_pair_kv_idces"],
            atom_single_mask=inputs["atom_exists_mask"],
            atom_token_indices=inputs["atom_token_index"].long(),
            crop_size=model_size,
            move_to_device=self.device,
            return_on_cpu=False,
        )

        # --- 6. Trunk Recycles ---
        logger.info(f"Running trunk ({num_trunk_recycles} recycles)")
        for _ in range(num_trunk_recycles):
            tok_single_curr, tok_pair_curr = self.trunk.forward(
                token_single_trunk_initial_repr=tok_single_curr,
                token_pair_trunk_initial_repr=tok_pair_curr,
                token_single_trunk_repr=tok_single_curr,
                token_pair_trunk_repr=tok_pair_curr,
                msa_input_feats=embedded["MSA"].to(self.dtype),
                msa_mask=inputs["msa_mask"],
                template_input_feats=embedded["TEMPLATES"].to(self.dtype),
                template_input_masks=und_self(inputs["template_mask"], "b t n1, b t n2 -> b t n1 n2"),
                token_single_mask=inputs["token_exists_mask"],
                token_pair_mask=und_self(inputs["token_exists_mask"], "b i, b j -> b i j"),
                crop_size=model_size,
                move_to_device=self.device,
            )

        # Slice to remove padding (Chai pads to crop sizes like 256, 384, 512)
        L = len(sequence)
        single_emb = tok_single_curr[0, :L].cpu().half()  # [L, D_single]
        pair_emb = tok_pair_curr[0, :L, :L].cpu().half()  # [L, L, D_pair]

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
    use_esm_embeddings: bool = False,
    skip_existing: bool = True,
) -> None:
    """
    Extract embeddings for all proteins and save to disk.

    Args:
        sequences: Dict of {protein_name: sequence}
        output_dir: Directory to save embeddings
        device: CUDA device
        num_trunk_recycles: Number of trunk recycles
        use_esm_embeddings: Whether to use ESM embeddings (default: False)
        skip_existing: Skip proteins that already have embeddings
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = ChaiTrunkExtractor(device=device)

    for name, seq in tqdm(sequences.items(), desc="Extracting embeddings"):
        out_path = output_dir / f"{name}.pt"

        if skip_existing and out_path.exists():
            logger.info(f"Skipping {name} (already exists)")
            continue

        try:
            embeddings = extractor.extract(
                sequence=seq,
                protein_name=name,
                num_trunk_recycles=num_trunk_recycles,
                use_esm_embeddings=use_esm_embeddings,
            )
            embeddings.save(out_path)
            logger.info(f"Saved {name} to {out_path}")

        except Exception as e:
            logger.error(f"Failed to extract {name}: {e}")
            continue


def worker(
    worker_id: int,
    total_workers: int,
    sequences: dict[str, str],
    output_dir: Path,
    device: str = "cuda:0",
    num_trunk_recycles: int = 3,
    use_esm_embeddings: bool = False,
) -> None:
    """Worker function for parallel processing across multiple GPUs/processes."""
    # Shard sequences for this worker
    all_keys = sorted(sequences.keys())
    my_keys = all_keys[worker_id::total_workers]
    logger.info(f"Worker {worker_id}: Processing {len(my_keys)}/{len(all_keys)} proteins")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = ChaiTrunkExtractor(device=device)

    for name in tqdm(my_keys, desc=f"Worker {worker_id}", position=worker_id):
        out_path = output_dir / f"{name}.pt"

        if out_path.exists():
            continue

        try:
            embeddings = extractor.extract(
                sequence=sequences[name],
                protein_name=name,
                num_trunk_recycles=num_trunk_recycles,
                use_esm_embeddings=use_esm_embeddings,
            )
            embeddings.save(out_path)
        except Exception as e:
            logger.error(f"Worker {worker_id} failed on {name}: {e}")


if __name__ == "__main__":
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
    parser.add_argument("--use-esm", action="store_true", help="Enable ESM embeddings (disabled by default)")
    parser.add_argument("--no-skip", action="store_true", help="Re-extract existing")
    parser.add_argument("--worker-id", type=int, default=0, help="Worker ID for parallel processing")
    parser.add_argument("--total-workers", type=int, default=1, help="Total number of workers")

    args = parser.parse_args()

    # Load sequences
    with open(args.sequences) as f:
        sequences = json.load(f)

    print(f"Loaded {len(sequences)} sequences")
    print(f"Output directory: {args.output_dir}")

    if args.total_workers > 1:
        worker(
            worker_id=args.worker_id,
            total_workers=args.total_workers,
            sequences=sequences,
            output_dir=args.output_dir,
            device=args.device,
            num_trunk_recycles=args.recycles,
            use_esm_embeddings=args.use_esm,
        )
    else:
        extract_all_proteins(
            sequences=sequences,
            output_dir=args.output_dir,
            device=args.device,
            num_trunk_recycles=args.recycles,
            use_esm_embeddings=args.use_esm,
            skip_existing=not args.no_skip,
        )
