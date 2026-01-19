"""Quick sanity check that dependencies are installed correctly."""

import torch
import numpy as np
import pandas as pd
from datasets import load_dataset

print("chai-thermo environment check")
print(f"  torch:   {torch.__version__}")
print(f"  numpy:   {np.__version__}")
print(f"  pandas:  {pd.__version__}")
print(f"  CUDA:    {torch.cuda.is_available()}")
print("\nAll dependencies OK!")
