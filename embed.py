import cv2
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description="Embed a 64x64 watermark into a 512x512 host image.")
parser.add_argument("--host", required=True, help="Host image, e.g. squirrel.png")
parser.add_argument("--wm", required=True, help="Watermark image, e.g. PTIT_logo.png or wm_text.png")
parser.add_argument("--alpha", type=int, default=8, help="Embedding strength")
parser.add_argument("--output", default=None, help="Output stego image name")
args = parser.parse_args()

host = cv2.imread(args.host, cv2.IMREAD_GRAYSCALE)
wm = cv2.imread(args.wm, cv2.IMREAD_GRAYSCALE)

if host is None:
    raise FileNotFoundError(f"Cannot read host image: {args.host}")
if wm is None:
    raise FileNotFoundError(f"Cannot read watermark image: {args.wm}")

if host.shape != (512, 512):
    raise ValueError(f"Host image must be 512x512, got {host.shape}")
if wm.shape != (64, 64):
    raise ValueError(f"Watermark image must be 64x64, got {wm.shape}")

wm_bin = (wm > 127).astype(np.uint8)
stego = host.astype(np.int16).copy()

for i in range(64):
    for j in range(64):
        x, y = i * 8, j * 8
        if wm_bin[i, j] == 1:
            stego[x, y] += args.alpha
        else:
            stego[x, y] -= args.alpha

stego = np.clip(stego, 0, 255).astype(np.uint8)

if args.output is None:
    host_name = os.path.splitext(os.path.basename(args.host))[0]
    wm_name = os.path.splitext(os.path.basename(args.wm))[0]
    output_name = f"{host_name}_wm_{wm_name}.png"
else:
    output_name = args.output

cv2.imwrite(output_name, stego)
print(f"Saved: {output_name}")
