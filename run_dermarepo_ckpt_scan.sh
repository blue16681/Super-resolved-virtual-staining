#!/usr/bin/env bash
set -euo pipefail

DATA_HE="dataset/E-Staining DermaRepo_cropped_split/test/HE"
DATA_IHC="dataset/E-Staining DermaRepo_cropped_split/test/IHC"
MODEL_DIR="models/BBDM-20260501-2108"
OUT_ROOT="output"
BATCH_SIZE="4"
AVG_STEPS="50"
SAMPLER="skip"

N=$(find "$DATA_HE" -maxdepth 1 -type f | wc -l)
echo "[INFO] dataset size: $N"

declare -a STEPS=(080000 090000 100000 110000)

for STEP in "${STEPS[@]}"; do
  CKPT="$MODEL_DIR/ema_0.9999_${STEP}_compress.pt"
  if [[ ! -f "$CKPT" ]]; then
    echo "[WARN] missing ckpt: $CKPT, skip"
    continue
  fi

  OUT_DIR="$OUT_ROOT/outputs_DermaRepo_test_png_skip50_step${STEP}_n1000"
  MODEL_NAME="DermaRepo_skip50_step${STEP}_n1000"
  VIS_DIR="$OUT_ROOT/visuals_DermaRepo_step${STEP}_n1000_top20"

  echo "[INFO] ===== Step $STEP start ====="
  echo "[INFO] out_dir: $OUT_DIR"

  conda run -n BBPM python test_bci_png.py \
    --model_path "$CKPT" \
    --lr_data_dir "$DATA_HE" \
    --hr_data_dir "$DATA_IHC" \
    --batch_size "$BATCH_SIZE" \
    --num_samples 1000 \
    --cond_channels 3 \
    --out_channels 3 \
    --sampler "$SAMPLER" \
    --avg_steps "$AVG_STEPS" \
    --large_size 256 \
    --crop_mode center \
    --save_dir "$OUT_DIR"

  conda run -n BBPM python analysis_metrics.py \
    --pred_dir "$OUT_DIR" \
    --model_name "$MODEL_NAME"

  # export first 20 triplets as quick visual set
  mkdir -p "$VIS_DIR"
  conda run -n BBPM python - <<PY
from pathlib import Path
import shutil
out_dir = Path(r"$OUT_DIR")
vis_dir = Path(r"$VIS_DIR")
preds = sorted([p for p in out_dir.glob('*.png') if not p.name.endswith('_target.png') and not p.name.endswith('_inp.png')], key=lambda p: int(p.stem))
for p in preds[:20]:
    t = out_dir / f"{p.stem}_target.png"
    i = out_dir / f"{p.stem}_inp.png"
    for src in [p, t, i]:
        if src.exists():
            shutil.copy2(src, vis_dir / src.name)
print(f"copied {min(20, len(preds))} samples to {vis_dir}")
PY

  echo "[INFO] ===== Step $STEP done ====="
done

echo "[INFO] all requested checkpoints completed"
