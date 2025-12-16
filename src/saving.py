import os
import numpy as np

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_npy(out_dir: str, name: str, arr):
    ensure_dir(out_dir)
    np.save(os.path.join(out_dir, name), arr, allow_pickle=True)

def save_text(out_dir: str, name: str, lines: list[str]):
    ensure_dir(out_dir)
    with open(os.path.join(out_dir, name), "w") as f:
        for line in lines:
            f.write(line + "\n")
