import copy
import csv
import functools
import os
from datetime import datetime

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
import time


import matplotlib.pyplot as plt
plt.ion()
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from . import dist_util, logger
from .fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0



class TrainLoop:
    def __init__(
        self,
        *,
        model,
        model_compressor,
        diffusion,
        data,
        val_data=None,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        val_interval,
        resume_checkpoint,
        model_dir="tmp",
        log_dir="log",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        val_num_samples=0,
        val_metrics_csv="val_metrics.csv",
        val_disable_lpips=False,
    ):
        self.model = model
        self.model_compression = model_compressor
        self.diffusion = diffusion
        self.data = data
        self.val_data = val_data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.val_interval = val_interval
        self.resume_checkpoint = resume_checkpoint
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.val_num_samples = val_num_samples
        self.val_metrics_csv = os.path.join(self.log_dir, val_metrics_csv)
        self.val_disable_lpips = val_disable_lpips
        self.val_lpips = None

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * 1

        self.model_params = list(self.model.parameters()) + list(self.model_compression.parameters())
        # add new parameters for CNN
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        if self.use_fp16:
            self._setup_fp16()

        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

        self.use_ddp = False
        self.ddp_model = self.model
        self.ddp_model_compress = self.model_compression

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            # if dist.get_rank() == 0:
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(
                dist_util.load_state_dict(
                    resume_checkpoint, map_location=dist_util.dev()
                )
            )
            self.model_compression.load_state_dict(
                dist_util.load_state_dict(
                    resume_checkpoint.replace('.pt','_compress.pt'), map_location=dist_util.dev()
                )
            )

        dist_util.sync_params(self.model.parameters())
        dist_util.sync_params(self.model_compression.parameters())

    
    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if 0 == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self._state_dict_to_master_params(state_dict)

                state_dict_compress = dist_util.load_state_dict(
                    ema_checkpoint.replace('.pt','_compress.pt'), map_location=dist_util.dev()
                )
                ema_params_compress = self._state_dict_to_master_params_compress(state_dict_compress)
        
        ema_params = ema_params + ema_params_compress

        dist_util.sync_params(ema_params)
        
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def _setup_fp16(self):
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()
        self.model_compression.convert_to_fp16()

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond, kwargs = next(self.data)

            self.run_step(batch, cond, kwargs)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.val_interval == 0 and self.step > 0:
                if self.val_data is not None:
                    self.val_step()
                else:
                    batch, cond, kwargs = next(self.data)
                    self.val_step_random_batch(batch, cond, kwargs)
            if self.step % self.save_interval == 0 and self.step > 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond, kwargs):
        self.forward_backward(batch, cond, kwargs)

        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()
        self.log_step()

    def val_step_random_batch(self, batch, cond, kwargs):
        self.model.eval()
        self.model_compression.eval()

        cond = cond.to(dist_util.dev())
        kwargs = {k: v.to(dist_util.dev()) for k, v in kwargs.items()}
        with th.no_grad():
            # add a compression CNN
            cond = self.model_compression(cond)
            sample = self.diffusion.p_sample_loop(
                self.model,
                cond,
                (self.batch_size, 3, batch.shape[-2], batch.shape[-1]),
                # clip_denoised=True,
                model_kwargs=kwargs,
            )
        cond = ((cond +1) * 127.5).clamp(0, 255).to(th.uint8)
        cond = cond.permute(0, 2, 3, 1).squeeze(-1)
        cond = cond.contiguous().cpu().numpy()
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1).squeeze(-1)
        sample = sample.contiguous().cpu().numpy()
        batch = ((batch +1) * 127.5).clamp(0, 255).to(th.uint8)
        batch = batch.permute(0, 2, 3, 1).squeeze(-1)
        batch = batch.contiguous().cpu().numpy()
        self.model.train()

        for i in range(self.batch_size):
            plt.imsave(os.path.join(self.log_dir, '%d_LR.png'%i), cond[i])
            plt.imsave(os.path.join(self.log_dir, '%d_SR.png'%i), sample[i])
            plt.imsave(os.path.join(self.log_dir, '%d_HR.png'%i), batch[i])

    def _build_lpips_metric(self):
        if self.val_disable_lpips or self.val_lpips is not None:
            return
        try:
            import lpips

            self.val_lpips = lpips.LPIPS(net="vgg").to(dist_util.dev())
            self.val_lpips.eval()
            logger.log("validation LPIPS enabled.")
        except Exception as e:
            self.val_lpips = None
            logger.log(f"validation LPIPS disabled due to init error: {e}")

    @staticmethod
    def _to_uint8_nhwc(tensor):
        arr = ((tensor + 1.0) * 127.5).clamp(0, 255).to(th.uint8)
        arr = arr.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        return arr

    def _append_val_metrics_csv(self, step, n_samples, psnr_mean, ssim_mean, lpips_mean, elapsed_sec):
        need_header = not os.path.exists(self.val_metrics_csv)
        with open(self.val_metrics_csv, "a", newline="") as f:
            writer = csv.writer(f)
            if need_header:
                writer.writerow(
                    ["timestamp", "step", "num_samples", "psnr_mean", "ssim_mean", "lpips_mean", "elapsed_sec"]
                )
            writer.writerow(
                [
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    int(step),
                    int(n_samples),
                    float(psnr_mean),
                    float(ssim_mean),
                    (float(lpips_mean) if lpips_mean is not None else ""),
                    float(elapsed_sec),
                ]
            )

    def val_step(self):
        self.model.eval()
        self.model_compression.eval()
        self._build_lpips_metric()

        max_samples = self.val_num_samples if self.val_num_samples > 0 else None
        total_psnr = 0.0
        total_ssim = 0.0
        total_lpips = 0.0
        total_count = 0
        t0 = time.time()

        if tqdm is not None:
            progress_total = None
            try:
                full_batches = len(self.val_data)
            except Exception:
                full_batches = None

            if max_samples is not None:
                bs = getattr(self.val_data, "batch_size", None)
                if bs is not None and bs > 0:
                    progress_total = int(np.ceil(max_samples / float(bs)))
                    if full_batches is not None:
                        progress_total = min(progress_total, full_batches)
                else:
                    progress_total = full_batches
            else:
                progress_total = full_batches

            val_iter = tqdm(
                self.val_data,
                total=progress_total,
                desc=f"[val] step {self.step + self.resume_step}",
                leave=False,
            )
        else:
            val_iter = self.val_data

        with th.no_grad():
            for item in val_iter:
                if len(item) < 3:
                    raise ValueError("Validation loader must return (target, input, kwargs).")

                batch, cond, kwargs = item[0], item[1], item[2]

                if max_samples is not None:
                    remain = max_samples - total_count
                    if remain <= 0:
                        break
                    if batch.shape[0] > remain:
                        batch = batch[:remain]
                        cond = cond[:remain]
                        if isinstance(kwargs, dict):
                            kwargs = {k: v[:remain] for k, v in kwargs.items()}

                batch = batch.to(dist_util.dev())
                cond = cond.to(dist_util.dev())
                if isinstance(kwargs, dict):
                    kwargs = {k: v.to(dist_util.dev()) for k, v in kwargs.items()}
                else:
                    kwargs = {}

                cond_comp = self.model_compression(cond)
                sample = self.diffusion.p_sample_loop(
                    self.model,
                    cond_comp,
                    (batch.shape[0], 3, batch.shape[-2], batch.shape[-1]),
                    model_kwargs=kwargs,
                )

                pred_u8 = self._to_uint8_nhwc(sample)
                gt_u8 = self._to_uint8_nhwc(batch)

                for i in range(pred_u8.shape[0]):
                    psnr = compare_psnr(gt_u8[i], pred_u8[i], data_range=255)
                    ssim = (
                        compare_ssim(gt_u8[i][:, :, 0], pred_u8[i][:, :, 0], data_range=255)
                        + compare_ssim(gt_u8[i][:, :, 1], pred_u8[i][:, :, 1], data_range=255)
                        + compare_ssim(gt_u8[i][:, :, 2], pred_u8[i][:, :, 2], data_range=255)
                    ) / 3.0
                    total_psnr += psnr
                    total_ssim += ssim

                if self.val_lpips is not None:
                    lpips_scores = self.val_lpips(sample, batch).view(-1).detach().cpu().numpy()
                    total_lpips += float(lpips_scores.sum())

                total_count += pred_u8.shape[0]
                if tqdm is not None:
                    val_iter.set_postfix(
                        samples=total_count,
                        psnr=f"{(total_psnr / total_count):.3f}",
                        ssim=f"{(total_ssim / total_count):.3f}",
                    )

        elapsed = time.time() - t0
        self.model.train()
        self.model_compression.train()

        if total_count == 0:
            logger.log("validation skipped: no samples evaluated.")
            return

        psnr_mean = total_psnr / total_count
        ssim_mean = total_ssim / total_count
        lpips_mean = (total_lpips / total_count) if self.val_lpips is not None else None

        logger.logkv("val_psnr", psnr_mean)
        logger.logkv("val_ssim", ssim_mean)
        if lpips_mean is not None:
            logger.logkv("val_lpips", lpips_mean)
        logger.logkv("val_samples", total_count)
        logger.logkv("val_time_sec", elapsed)
        logger.dumpkvs()

        self._append_val_metrics_csv(
            step=self.step + self.resume_step,
            n_samples=total_count,
            psnr_mean=psnr_mean,
            ssim_mean=ssim_mean,
            lpips_mean=lpips_mean,
            elapsed_sec=elapsed,
        )
        logger.log(
            f"validation done @ step {self.step + self.resume_step}: "
            f"samples={total_count}, PSNR={psnr_mean:.4f}, SSIM={ssim_mean:.4f}, "
            f"LPIPS={lpips_mean if lpips_mean is not None else 'N/A'}"
        )

    def forward_backward(self, batch, cond, kwargs):
        zero_grad(self.model_params)
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = cond[i : i + self.microbatch].to(dist_util.dev())
            # add a compression CNN
            micro_cond = self.ddp_model_compress(micro_cond)
            micro_kwargs = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in kwargs.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                micro_cond,
                t,
                model_kwargs=micro_kwargs,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            if self.use_fp16:
                loss_scale = 2 ** self.lg_loss_scale
                (loss * loss_scale).backward()
            else:
                loss.backward()

    def optimize_fp16(self):
        if any(not th.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return

        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def optimize_normal(self):
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        sqsum = 0.0
        for p in self.master_params:
            sqsum += (p.grad ** 2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        if self.use_fp16:
            logger.logkv("lg_loss_scale", self.lg_loss_scale)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            state_dict_compress = self._master_params_to_state_dict_compress(params)
            if 0 == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                    filename_compress = f"model{(self.step+self.resume_step):06d}_compress.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                    filename_compress = f"ema_{rate}_{(self.step+self.resume_step):06d}_compress.pt"
                with bf.BlobFile(bf.join(self.model_dir, filename), "wb") as f:
                    th.save(state_dict, f)
                with bf.BlobFile(bf.join(self.model_dir, filename_compress), "wb") as f:
                    th.save(state_dict_compress, f)

        save_checkpoint(0, self.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if 0 == 0:
            with bf.BlobFile(
                bf.join(self.model_dir, f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

    def _master_params_to_state_dict(self, master_params):
        if self.use_fp16:
            master_params = unflatten_master_params(
                list(self.model.parameters()) + list(self.model_compression.parameters()), master_params
            )
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict
    
    def _master_params_to_state_dict_compress(self, master_params):
        if self.use_fp16:
            master_params = unflatten_master_params(
                list(self.model.parameters()) + list(self.model_compression.parameters()), master_params
            )
        state_dict = self.model_compression.state_dict()
        for i, (name, _value) in enumerate(self.model_compression.named_parameters()):
            assert name in state_dict
            # state_dict[name] = master_params[-6+i]
            state_dict[name] = master_params[456+i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        if self.use_fp16:
            return make_master_params(params)
        else:
            return params
    
        
    def _state_dict_to_master_params_compress(self, state_dict):
        params = [state_dict[name] for name, _ in self.model_compression.named_parameters()]
        if self.use_fp16:
            return make_master_params(params)
        else:
            return params


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir())


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
