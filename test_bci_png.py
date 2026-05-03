import argparse
import glob
import os

import numpy as np
import torch as th
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from improved_diffusion import dist_util, logger
from improved_diffusion.unet import CompressCNN_upsample as CompressCNN
from improved_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)


class PairedPNGDataset(Dataset):
    def __init__(self, input_dir, target_dir, image_size=256, crop_mode="center"):
        super().__init__()
        self.image_size = image_size
        self.crop_mode = crop_mode

        input_files = sorted(
            [p for p in glob.glob(os.path.join(input_dir, "*")) if os.path.isfile(p)]
        )
        target_files = sorted(
            [p for p in glob.glob(os.path.join(target_dir, "*")) if os.path.isfile(p)]
        )

        input_map = {os.path.basename(p): p for p in input_files}
        target_map = {os.path.basename(p): p for p in target_files}
        self.names = sorted(list(set(input_map.keys()) & set(target_map.keys())))

        if not self.names:
            raise ValueError(f"No paired files found in {input_dir} and {target_dir}")

        self.input_paths = [input_map[n] for n in self.names]
        self.target_paths = [target_map[n] for n in self.names]

    def __len__(self):
        return len(self.names)

    def _crop_pair(self, inp, tag):
        h, w = inp.shape[:2]
        if h < self.image_size or w < self.image_size:
            raise ValueError(
                f"Image too small for crop: {(h, w)} < {self.image_size}. "
                "Please lower --large_size or provide larger test images."
            )

        if h == self.image_size and w == self.image_size:
            return inp, tag

        if self.crop_mode == "center":
            y0 = (h - self.image_size) // 2
            x0 = (w - self.image_size) // 2
        elif self.crop_mode == "top_left":
            y0, x0 = 0, 0
        else:
            raise ValueError(f"Unsupported crop_mode: {self.crop_mode}")

        y1, x1 = y0 + self.image_size, x0 + self.image_size
        return inp[y0:y1, x0:x1], tag[y0:y1, x0:x1]

    def __getitem__(self, idx):
        inp = np.array(Image.open(self.input_paths[idx]).convert("RGB"), dtype=np.float32)
        tag = np.array(Image.open(self.target_paths[idx]).convert("RGB"), dtype=np.float32) / 255.0

        if inp.shape[:2] != tag.shape[:2]:
            raise ValueError(
                f"Shape mismatch for {self.names[idx]}: HE {inp.shape} vs IHC {tag.shape}"
            )

        inp, tag = self._crop_pair(inp, tag)

        inp = (inp - inp.mean()) / (inp.std() + 1e-6)
        tag = tag * 255.0 / 127.5 - 1.0

        inp = inp.transpose(2, 0, 1)
        tag = tag.transpose(2, 0, 1)
        return th.from_numpy(tag).float(), th.from_numpy(inp).float(), self.names[idx]


def min_max_norm(img):
    vmin = img.min()
    vmax = img.max()
    if vmax - vmin < 1e-8:
        return np.zeros_like(img)
    return (img - vmin) / (vmax - vmin)


def to_uint8_rgb(img_nchw):
    img = ((img_nchw + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return img


def create_argparser():
    defaults = dict(
        lr_data_dir="dataset/BCIdataset/test/HE",
        hr_data_dir="dataset/BCIdataset/test/IHC",
        model_path="",
        save_dir="outputs_bci_test_png",
        cond_channels=3,
        out_channels=3,
        batch_size=1,
        num_samples=0,
        num_workers=0,
        sampler="skip",
        avg_steps=50,
        clip_denoised=False,
        crop_mode="center",
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def sample_with_strategy(diffusion, model, cond, args, model_kwargs):
    shape = (cond.shape[0], 3, args.large_size, args.large_size)
    if args.sampler == "mean":
        sample = diffusion.p_sample_mean_loop(
            model,
            cond,
            n_avg_steps=args.avg_steps,
            shape=shape,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
    elif args.sampler == "standard":
        sample = diffusion.p_sample_loop(
            model,
            cond,
            shape=shape,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
    elif args.sampler == "skip":
        sample = diffusion.p_sample_mean_skip(
            model,
            cond,
            n_avg_steps=args.avg_steps,
            shape=shape,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
    else:
        raise ValueError(f"Unknown sampler: {args.sampler}")
    return sample


def main():
    args = create_argparser().parse_args()

    if not args.model_path:
        raise ValueError("Please provide --model_path (recommend *_compress.pt)")

    if args.model_path.endswith("_compress.pt"):
        model_path_compress = args.model_path
        model_path = args.model_path.replace("_compress.pt", ".pt")
    else:
        model_path = args.model_path
        model_path_compress = args.model_path.replace(".pt", "_compress.pt")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Main diffusion checkpoint not found: {model_path}")
    if not os.path.exists(model_path_compress):
        raise FileNotFoundError(f"Compressor checkpoint not found: {model_path_compress}")

    os.makedirs(args.save_dir, exist_ok=True)

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys()),
        in_channels=3,
        out_channels=3,
    )
    model.load_state_dict(dist_util.load_state_dict(model_path, map_location="cpu"))
    model.to(dist_util.dev())
    model.eval()

    model_compressor = CompressCNN(args.cond_channels, args.out_channels, dims=2)
    model_compressor.load_state_dict(
        dist_util.load_state_dict(model_path_compress, map_location="cpu")
    )
    model_compressor.to(dist_util.dev())
    model_compressor.eval()

    logger.log(f"loaded model: {model_path}")
    logger.log(f"loaded compressor: {model_path_compress}")

    dataset = PairedPNGDataset(
        input_dir=args.lr_data_dir,
        target_dir=args.hr_data_dir,
        image_size=args.large_size,
        crop_mode=args.crop_mode,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    max_samples = len(dataset) if args.num_samples <= 0 else min(args.num_samples, len(dataset))
    logger.log(f"running inference for {max_samples} samples")

    saved = 0
    with th.no_grad():
        for batch, cond, names in loader:
            if saved >= max_samples:
                break

            b = min(batch.shape[0], max_samples - saved)
            batch = batch[:b].to(dist_util.dev())
            cond = cond[:b].to(dist_util.dev())
            names = names[:b]

            cond_comp = model_compressor(cond)
            sample = sample_with_strategy(diffusion, model, cond_comp, args, model_kwargs={})

            pred = sample.permute(0, 2, 3, 1).contiguous().cpu().numpy()
            gt = batch.permute(0, 2, 3, 1).contiguous().cpu().numpy()
            inp = cond_comp.permute(0, 2, 3, 1).contiguous().cpu().numpy()

            pred = to_uint8_rgb(pred)
            gt = to_uint8_rgb(gt)

            for i in range(b):
                base = os.path.splitext(names[i])[0]

                pred_path = os.path.join(args.save_dir, f"{base}.png")
                target_path = os.path.join(args.save_dir, f"{base}_target.png")
                inp_path = os.path.join(args.save_dir, f"{base}_inp.png")

                plt.imsave(pred_path, pred[i])
                plt.imsave(target_path, gt[i])
                plt.imsave(inp_path, min_max_norm(inp[i]))
                saved += 1

            logger.log(f"saved {saved}/{max_samples}")

    logger.log(f"done. results saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
