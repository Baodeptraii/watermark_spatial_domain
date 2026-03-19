import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Convert text in message.txt to a 64x64 binary watermark image.")
parser.add_argument("--message", default="message.txt", help="Input text file")
parser.add_argument("--output", default="wm_text.png", help="Output watermark image")
parser.add_argument("--size", type=int, default=64, help="Watermark size, default 64")
args = parser.parse_args()

with open(args.message, "r", encoding="utf-8") as f:
    msg = f.read()

if len(msg) == 0:
    raise ValueError("Message file is empty")

bits = ''.join(format(ord(c), '08b') for c in msg)

wm = np.zeros((args.size, args.size), dtype=np.uint8)

for i in range(args.size):
    for j in range(args.size):
        bit = bits[(i * args.size + j) % len(bits)]
        wm[i, j] = 255 if bit == '1' else 0

cv2.imwrite(args.output, wm)
print(f"Saved: {args.output}")
