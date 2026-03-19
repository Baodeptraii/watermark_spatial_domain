import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Calculate PSNR between two grayscale images.")
parser.add_argument("--img1", required=True, help="Original image")
parser.add_argument("--img2", required=True, help="Modified image")
args = parser.parse_args()

img1 = cv2.imread(args.img1, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(args.img2, cv2.IMREAD_GRAYSCALE)

if img1 is None:
    raise FileNotFoundError(f"Cannot read image: {args.img1}")
if img2 is None:
    raise FileNotFoundError(f"Cannot read image: {args.img2}")

if img1.shape != img2.shape:
    raise ValueError(f"Images must have the same shape, got {img1.shape} and {img2.shape}")

mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)

if mse == 0:
    print("PSNR: inf dB")
else:
    psnr = 10 * np.log10((255 ** 2) / mse)
    print(f"PSNR: {psnr:.4f} dB")
