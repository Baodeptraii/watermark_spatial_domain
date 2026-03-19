import cv2
import numpy as np
import argparse
import os


def embed_watermark_lsb_block(host: np.ndarray, wm: np.ndarray) -> np.ndarray:
    """
    Nhúng watermark grayscale 64x64 vào ảnh host 512x512 theo kiểu block 8x8.
    Mỗi pixel watermark (8 bit) được ghi vào 8 pixel đầu tiên của hàng đầu block 8x8.

    - Host: 512x512 grayscale
    - Watermark: 64x64 grayscale
    - Stego đầu ra vẫn là PNG lossless, có thể trích xuất lại watermark gần như/chính xác tuyệt đối.
    """
    if host.shape != (512, 512):
        raise ValueError(f"Host image must be 512x512, got {host.shape}")
    if wm.shape != (64, 64):
        raise ValueError(f"Watermark image must be 64x64, got {wm.shape}")

    stego = host.copy()
    block_size = 8

    for i in range(64):
        for j in range(64):
            x = i * block_size
            y = j * block_size

            wm_val = int(wm[i, j])
            bits = f"{wm_val:08b}"

            # Ghi 8 bit watermark vào 8 pixel đầu tiên của block
            for k, bit in enumerate(bits):
                px = int(stego[x, y + k])
                stego[x, y + k] = (px & 0xFE) | int(bit)

    return stego


parser = argparse.ArgumentParser(
    description="Embed a 64x64 grayscale watermark into a 512x512 host image using block-based LSB."
)
parser.add_argument("--host", required=True, help="Host image, e.g. squirrel.png")
parser.add_argument("--wm", required=True, help="Watermark image, e.g. PTIT_logo.png or wm_text.png")
parser.add_argument("--output", default=None, help="Output stego image name")
args = parser.parse_args()

host = cv2.imread(args.host, cv2.IMREAD_GRAYSCALE)
wm = cv2.imread(args.wm, cv2.IMREAD_GRAYSCALE)

if host is None:
    raise FileNotFoundError(f"Cannot read host image: {args.host}")
if wm is None:
    raise FileNotFoundError(f"Cannot read watermark image: {args.wm}")

stego = embed_watermark_lsb_block(host, wm)

if args.output is None:
    host_name = os.path.splitext(os.path.basename(args.host))[0]
    wm_name = os.path.splitext(os.path.basename(args.wm))[0]
    output_name = f"{host_name}_wm2_{wm_name}.png"
else:
    output_name = args.output

ok = cv2.imwrite(output_name, stego)
if not ok:
    raise IOError(f"Cannot write output image: {output_name}")

print(f"Saved: {output_name}")
print("Method: block-based 8-LSB embedding (1 watermark pixel / 1 host block)")