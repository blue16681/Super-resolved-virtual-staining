import argparse
import csv
import glob
import os

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


def evaluate_images(output_path, target_path):
    output_img = np.array(Image.open(output_path).convert("RGB"))
    target_img = np.array(Image.open(target_path).convert("RGB"))

    psnr = compare_psnr(target_img, output_img, data_range=255)
    ssim = (
        compare_ssim(target_img[:, :, 0], output_img[:, :, 0], data_range=255)
        + compare_ssim(target_img[:, :, 1], output_img[:, :, 1], data_range=255)
        + compare_ssim(target_img[:, :, 2], output_img[:, :, 2], data_range=255)
    ) / 3.0

    return psnr, ssim, output_img, target_img


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", type=str, required=True, help="directory with predicted images")
    parser.add_argument("--pred_suffix", type=str, default=".png", help="prediction suffix")
    parser.add_argument("--target_suffix", type=str, default="_target.png", help="GT suffix in pred_dir")
    parser.add_argument("--exclude_suffix", type=str, default="_inp.png", help="suffix to ignore")
    parser.add_argument("--csv_name", type=str, default="metrics.csv", help="output csv file name")
    parser.add_argument("--disable_lpips", action="store_true", help="disable LPIPS")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="device for LPIPS")
    return parser


def list_prediction_files(pred_dir, pred_suffix, target_suffix, exclude_suffix):
    all_files = sorted(glob.glob(os.path.join(pred_dir, f"*{pred_suffix}")))
    preds = []
    for p in all_files:
        if p.endswith(target_suffix):
            continue
        if exclude_suffix and p.endswith(exclude_suffix):
            continue
        preds.append(p)
    return preds


def maybe_create_lpips(disable_lpips, device_choice):
    if disable_lpips:
        return None, None

    try:
        import torch
        import lpips
    except Exception as e:
        print(f"[WARN] LPIPS disabled (import failed): {e}")
        return None, None

    if device_choice == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_choice

    metric = lpips.LPIPS(net="vgg").to(device)
    return metric, device


def main():
    args = create_argparser().parse_args()

    if not os.path.isdir(args.pred_dir):
        raise FileNotFoundError(f"pred_dir does not exist: {args.pred_dir}")

    pred_files = list_prediction_files(
        args.pred_dir, args.pred_suffix, args.target_suffix, args.exclude_suffix
    )

    if not pred_files:
        raise ValueError(f"No prediction files found in: {args.pred_dir}")

    lpips_metric, lpips_device = maybe_create_lpips(args.disable_lpips, args.device)

    if lpips_metric is not None:
        import torch
        from torchvision import transforms

        to_tensor = transforms.ToTensor()

    rows = []
    missing_targets = 0

    for pred_path in pred_files:
        stem = pred_path[: -len(args.pred_suffix)]
        target_path = stem + args.target_suffix

        if not os.path.exists(target_path):
            missing_targets += 1
            continue

        psnr, ssim, pred_img, target_img = evaluate_images(pred_path, target_path)

        lpips_score = None
        if lpips_metric is not None:
            pred_t = to_tensor(Image.fromarray(pred_img)).unsqueeze(0).to(lpips_device)
            target_t = to_tensor(Image.fromarray(target_img)).unsqueeze(0).to(lpips_device)
            with torch.no_grad():
                lpips_score = lpips_metric(pred_t, target_t).mean().item()

        rows.append(
            {
                "name": os.path.basename(pred_path),
                "target": os.path.basename(target_path),
                "psnr": psnr,
                "ssim": ssim,
                "lpips": lpips_score,
            }
        )

    if not rows:
        raise ValueError("No valid prediction-target pairs found.")

    psnr_mean = float(np.mean([r["psnr"] for r in rows]))
    ssim_mean = float(np.mean([r["ssim"] for r in rows]))

    lpips_values = [r["lpips"] for r in rows if r["lpips"] is not None]
    lpips_mean = float(np.mean(lpips_values)) if lpips_values else None

    csv_path = os.path.join(args.pred_dir, args.csv_name)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "target", "psnr", "ssim", "lpips"])
        for r in rows:
            writer.writerow([r["name"], r["target"], r["psnr"], r["ssim"], r["lpips"]])

        writer.writerow([])
        writer.writerow(["mean", "-", psnr_mean, ssim_mean, lpips_mean])

    print(f"[INFO] evaluated pairs: {len(rows)}")
    if missing_targets > 0:
        print(f"[WARN] missing targets: {missing_targets}")
    print(f"[INFO] PSNR mean : {psnr_mean:.4f}")
    print(f"[INFO] SSIM mean : {ssim_mean:.4f}")
    if lpips_mean is None:
        print("[INFO] LPIPS mean: N/A (disabled or unavailable)")
    else:
        print(f"[INFO] LPIPS mean: {lpips_mean:.4f}")
    print(f"[INFO] saved csv : {csv_path}")


if __name__ == "__main__":
    main()
