import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Calculate NCC between original and extracted watermark.")
parser.add_argument("--wm1", required=True, help="Original watermark image")
parser.add_argument("--wm2", required=True, help="Extracted watermark image")
args = parser.parse_args()

wm1 = cv2.imread(args.wm1, cv2.IMREAD_GRAYSCALE)
wm2 = cv2.imread(args.wm2, cv2.IMREAD_GRAYSCALE)

if wm1 is None:
    raise FileNotFoundError(f"Cannot read watermark image: {args.wm1}")
if wm2 is None:
    raise FileNotFoundError(f"Cannot read watermark image: {args.wm2}")

if wm1.shape != wm2.shape:
    raise ValueError(f"Watermark images must have the same shape, got {wm1.shape} and {wm2.shape}")

a = wm1.astype(np.float64).flatten()
b = wm2.astype(np.float64).flatten()

if np.std(a) == 0 or np.std(b) == 0:
    raise ValueError("NCC is undefined because one watermark has zero variance")

ncc = np.corrcoef(a, b)[0, 1]
print(f"NCC: {ncc:.6f}")
