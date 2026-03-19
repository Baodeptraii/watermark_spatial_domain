import cv2
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description="Extract a 64x64 watermark from a stego image using the original host.")
parser.add_argument("--host", required=True, help="Original host image, e.g. squirrel.png")
parser.add_argument("--stego", required=True, help="Stego image, e.g. squirrel_wm_PTIT_logo.png")
parser.add_argument("--output", default=None, help="Output extracted watermark name")
args = parser.parse_args()

host = cv2.imread(args.host, cv2.IMREAD_GRAYSCALE)
stego = cv2.imread(args.stego, cv2.IMREAD_GRAYSCALE)

if host is None:
    raise FileNotFoundError(f"Cannot read host image: {args.host}")
if stego is None:
    raise FileNotFoundError(f"Cannot read stego image: {args.stego}")

if host.shape != (512, 512):
    raise ValueError(f"Host image must be 512x512, got {host.shape}")
if stego.shape != (512, 512):
    raise ValueError(f"Stego image must be 512x512, got {stego.shape}")

wm_ext = np.zeros((64, 64), dtype=np.uint8)

for i in range(64):
    for j in range(64):
        x, y = i * 8, j * 8
        diff = int(stego[x, y]) - int(host[x, y])
        wm_ext[i, j] = 255 if diff > 0 else 0

if args.output is None:
    stego_name = os.path.splitext(os.path.basename(args.stego))[0]
    output_name = f"{stego_name}_extracted.png"
else:
    output_name = args.output

cv2.imwrite(output_name, wm_ext)
print(f"Saved: {output_name}")
