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
    return psnr, ssim


def extract_slide_fov_names(path):
    """
    Extract slide name and FOV name from the path.
    Assumes the format `.../slide_name/fov_name_xx.png`.
    """
    parts = os.path.normpath(path).split(os.sep)
    slide_name = parts[-2]
    fov_name = os.path.splitext(parts[-1])[0]
    return slide_name, fov_name


def _resolve_device(device):
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _build_lpips_metric(disable_lpips=False, device="auto"):
    if disable_lpips:
        return None, None, None
    try:
        import lpips
        from torchvision import transforms

        dev = _resolve_device(device)
        metric = lpips.LPIPS(net="vgg").to(dev)
        return metric, transforms.ToTensor(), dev
    except Exception as e:
        print(f"[WARN] LPIPS disabled due to import/init error: {e}")
        return None, None, None


def compute_metrics_for_model(
    model_paths,
    gt_paths,
    model_name,
    output_dir,
    lpips_vgg=None,
    lpips_to_tensor=None,
    lpips_device="cpu",
):
    all_metrics = []
    n = min(len(model_paths), len(gt_paths))
    if len(model_paths) != len(gt_paths):
        print(
            f"[WARN] Pair length mismatch for {model_name}: "
            f"pred={len(model_paths)}, gt={len(gt_paths)}. Using first {n} pairs."
        )

    for model_path, gt_path in zip(model_paths[:n], gt_paths[:n]):
        output_img = Image.open(model_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")
        psnr, ssim = evaluate_images(model_path, gt_path)

        lpips_score = None
        if lpips_vgg is not None:
            import torch

            output_tensor = lpips_to_tensor(output_img).unsqueeze(0).to(lpips_device)
            gt_tensor = lpips_to_tensor(gt_img).unsqueeze(0).to(lpips_device)
            with torch.no_grad():
                lpips_score = lpips_vgg(output_tensor, gt_tensor).mean().item()

        slide_name_gt, fov_name_gt = extract_slide_fov_names(gt_path)
        slide_name, fov_name = extract_slide_fov_names(model_path)

        fov_name = fov_name.replace("_gt", "").replace("_target", "")
        fov_name_gt = fov_name_gt.replace("_gt", "").replace("_target", "")

        if slide_name != slide_name_gt or fov_name != fov_name_gt:
            print(
                f"[WARN] Name mismatch: pred={os.path.basename(model_path)} "
                f"gt={os.path.basename(gt_path)}"
            )

        all_metrics.append([slide_name, fov_name, psnr, ssim, lpips_score])

    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, f"{model_name}_metrics.csv")
    with open(csv_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Slide Name", "FOV Name", "PSNR", "SSIM", "LPIPS"])
        writer.writerows(all_metrics)
        writer.writerow([])
        writer.writerow(
            [
                "mean",
                "-",
                float(np.mean([x[2] for x in all_metrics])) if all_metrics else None,
                float(np.mean([x[3] for x in all_metrics])) if all_metrics else None,
                (
                    float(np.mean([x[4] for x in all_metrics if x[4] is not None]))
                    if any(x[4] is not None for x in all_metrics)
                    else None
                ),
            ]
        )
    print(f"[INFO] Metrics saved for model: {model_name} -> {csv_file}")
    return all_metrics


def _collect_pairs_single_dir(pred_dir, pred_suffix, target_suffix, exclude_suffix):
    pred_files = sorted(glob.glob(os.path.join(pred_dir, f"*{pred_suffix}")))
    preds, gts = [], []
    for p in pred_files:
        if p.endswith(target_suffix):
            continue
        if exclude_suffix and p.endswith(exclude_suffix):
            continue
        tgt = p[: -len(pred_suffix)] + target_suffix
        if os.path.exists(tgt):
            preds.append(p)
            gts.append(tgt)
    return preds, gts


def _collect_pairs_two_dirs(pred_dir, gt_dir, pred_suffix, gt_suffix):
    pred_files = sorted(glob.glob(os.path.join(pred_dir, f"*{pred_suffix}")))
    gt_map = {
        os.path.basename(p): p for p in glob.glob(os.path.join(gt_dir, f"*{gt_suffix}"))
    }
    preds, gts = [], []
    for p in pred_files:
        name = os.path.basename(p)
        if name in gt_map:
            preds.append(p)
            gts.append(gt_map[name])
    return preds, gts


def run_legacy_windows_paths():
    path_gt = sorted(glob.glob(r"I:\sr_vs_revision\outputs_1x_diff\*\*_gt.png"))
    model_paths = {
        "diffusion_1x": sorted(
            [path.replace("_gt.png", ".png") for path in glob.glob(r"I:\sr_vs_revision\outputs_1x_diff\*\*_gt.png")]
        ),
        "diffusion_2x": sorted(
            [path.replace("_gt.png", ".png") for path in glob.glob(r"I:\sr_vs_revision\outputs_2x_diff\*\*_gt.png")]
        ),
        "diffusion_3x": sorted(
            [path.replace("_gt.png", ".png") for path in glob.glob(r"I:\sr_vs_revision\outputs_3x_diff\*\*_gt.png")]
        ),
        "diffusion_4x": sorted(
            [path.replace("_gt.png", ".png") for path in glob.glob(r"I:\sr_vs_revision\outputs_4x_diff\*\*_gt.png")]
        ),
        "diffusion_5x": sorted(
            [path.replace("_gt.png", ".png") for path in glob.glob(r"I:\sr_vs_revision\outputs_5x_diff\*\*_gt.png")]
        ),
        "cgan_1x": sorted(
            [path.replace("_gt.png", ".png") for path in glob.glob(r"I:\sr_vs_revision\outputs_1x_cgan\*\*_gt.png")]
        ),
        "cgan_2x": sorted(
            [path.replace("_gt.png", ".png") for path in glob.glob(r"I:\sr_vs_revision\outputs_2x_cgan\*\*_gt.png")]
        ),
        "cgan_3x": sorted(
            [path.replace("_gt.png", ".png") for path in glob.glob(r"I:\sr_vs_revision\outputs_3x_cgan\*\*_gt.png")]
        ),
        "cgan_4x": sorted(
            [path.replace("_gt.png", ".png") for path in glob.glob(r"I:\sr_vs_revision\outputs_4x_cgan\*\*_gt.png")]
        ),
        "cgan_5x": sorted(
            [path.replace("_gt.png", ".png") for path in glob.glob(r"I:\sr_vs_revision\outputs_5x_cgan\*\*_gt.png")]
        ),
    }
    output_dir = r"I:\BBDM_model_tree_revision\analysis_revision_results\all_lung_metrics"

    lpips_vgg, to_tensor, lpips_device = _build_lpips_metric(disable_lpips=False, device="auto")
    for model_name, paths in model_paths.items():
        compute_metrics_for_model(
            paths,
            path_gt,
            model_name,
            output_dir,
            lpips_vgg=lpips_vgg,
            lpips_to_tensor=to_tensor,
            lpips_device=lpips_device,
        )


def create_argparser():
    # xxx_inp.png 输入条件图HE（condition）的可视化,不用与指标
    # xxx_target.png groundtruth，IHC
    # 
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", type=str, default="", help="Prediction image directory.")
    parser.add_argument("--gt_dir", type=str, default="", help="Optional GT directory with same filenames as predictions.")
    parser.add_argument("--output_dir", type=str, default="", help="Directory to save metric CSV.")
    parser.add_argument("--model_name", type=str, default="experiment", help="CSV prefix name.")
    parser.add_argument("--pred_suffix", type=str, default=".png", help="Prediction filename suffix.")
    parser.add_argument("--gt_suffix", type=str, default=".png", help="GT filename suffix when --gt_dir is used.")
    parser.add_argument("--target_suffix", type=str, default="_target.png", help="GT suffix in same directory mode.")
    parser.add_argument("--exclude_suffix", type=str, default="_inp.png", help="Ignored suffix in same directory mode.")
    parser.add_argument("--disable_lpips", action="store_true", help="Disable LPIPS calculation.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device for LPIPS.")
    parser.add_argument(
        "--legacy_windows_paths",
        action="store_true",
        help="Run the original hard-coded Windows path evaluation logic.",
    )
    return parser


def main():
    args = create_argparser().parse_args()

    if args.legacy_windows_paths:
        run_legacy_windows_paths()
        return

    if not args.pred_dir:
        raise ValueError("Please provide --pred_dir, or use --legacy_windows_paths.")

    if not os.path.isdir(args.pred_dir):
        raise FileNotFoundError(f"pred_dir not found: {args.pred_dir}")

    if args.gt_dir:
        if not os.path.isdir(args.gt_dir):
            raise FileNotFoundError(f"gt_dir not found: {args.gt_dir}")
        model_paths, gt_paths = _collect_pairs_two_dirs(
            args.pred_dir, args.gt_dir, args.pred_suffix, args.gt_suffix
        )
    else:
        model_paths, gt_paths = _collect_pairs_single_dir(
            args.pred_dir, args.pred_suffix, args.target_suffix, args.exclude_suffix
        )

    if not model_paths:
        raise ValueError("No valid prediction-target pairs found.")

    output_dir = args.output_dir if args.output_dir else args.pred_dir
    lpips_vgg, to_tensor, lpips_device = _build_lpips_metric(
        disable_lpips=args.disable_lpips, device=args.device
    )
    all_metrics = compute_metrics_for_model(
        model_paths,
        gt_paths,
        args.model_name,
        output_dir,
        lpips_vgg=lpips_vgg,
        lpips_to_tensor=to_tensor,
        lpips_device=lpips_device,
    )

    psnr_mean = float(np.mean([x[2] for x in all_metrics]))
    ssim_mean = float(np.mean([x[3] for x in all_metrics]))
    lpips_vals = [x[4] for x in all_metrics if x[4] is not None]
    lpips_mean = float(np.mean(lpips_vals)) if lpips_vals else None

    print(f"[INFO] evaluated pairs: {len(all_metrics)}")
    print(f"[INFO] PSNR mean: {psnr_mean:.4f}")
    print(f"[INFO] SSIM mean: {ssim_mean:.4f}")
    if lpips_mean is None:
        print("[INFO] LPIPS mean: N/A")
    else:
        print(f"[INFO] LPIPS mean: {lpips_mean:.4f}")


if __name__ == "__main__":
    main()
