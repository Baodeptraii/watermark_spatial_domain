import cv2
import numpy as np
import argparse
import os


def extract_watermark_lsb_block(stego: np.ndarray) -> np.ndarray:
    """
    Trích xuất lại watermark grayscale 64x64 từ ảnh stego 512x512.
    Đọc 8 LSB của 8 pixel đầu tiên trong mỗi block 8x8 để dựng lại 1 pixel watermark.

    Lưu ý: Với thiết kế embed2.py hiện tại, việc trích xuất không cần host gốc,
    nhưng tham số --host vẫn được giữ để tương thích giao diện cũ của bài lab.
    """
    if stego.shape != (512, 512):
        raise ValueError(f"Stego image must be 512x512, got {stego.shape}")

    wm_ext = np.zeros((64, 64), dtype=np.uint8)
    block_size = 8

    for i in range(64):
        for j in range(64):
            x = i * block_size
            y = j * block_size

            bits = []
            for k in range(8):
                bits.append(str(int(stego[x, y + k]) & 1))

            wm_ext[i, j] = int("".join(bits), 2)

    return wm_ext


parser = argparse.ArgumentParser(
    description="Extract a 64x64 grayscale watermark from a stego image using block-based LSB."
)
parser.add_argument("--host", required=False, default=None,
                    help="Original host image (optional, kept only for compatibility)")
parser.add_argument("--stego", required=True, help="Stego image, e.g. squirrel_wm2_PTIT_logo.png")
parser.add_argument("--output", default=None, help="Output extracted watermark name")
args = parser.parse_args()

if args.host is not None:
    host = cv2.imread(args.host, cv2.IMREAD_GRAYSCALE)
    if host is None:
        raise FileNotFoundError(f"Cannot read host image: {args.host}")
    if host.shape != (512, 512):
        raise ValueError(f"Host image must be 512x512, got {host.shape}")

stego = cv2.imread(args.stego, cv2.IMREAD_GRAYSCALE)
if stego is None:
    raise FileNotFoundError(f"Cannot read stego image: {args.stego}")

wm_ext = extract_watermark_lsb_block(stego)

if args.output is None:
    stego_name = os.path.splitext(os.path.basename(args.stego))[0]
    output_name = f"{stego_name}_extracted.png"
else:
    output_name = args.output

ok = cv2.imwrite(output_name, wm_ext)
if not ok:
    raise IOError(f"Cannot write output image: {output_name}")

print(f"Saved: {output_name}")
print("Method: block-based 8-LSB extraction")