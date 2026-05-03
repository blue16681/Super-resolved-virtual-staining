from pathlib import Path
import csv
import re

root = Path('output')
rows = []
pat = re.compile(r'DermaRepo_skip50_step(\d{6})_n(\d+)_metrics\.csv$')
for f in sorted(root.glob('**/*_metrics.csv')):
    m = pat.search(f.name)
    if not m:
        continue
    step = int(m.group(1))
    n = int(m.group(2))
    with open(f, newline='') as fp:
        rd = csv.reader(fp)
        data = list(rd)
    mean_row = None
    for r in data[::-1]:
        if len(r) >= 5 and r[0] == 'mean':
            mean_row = r
            break
    if mean_row is None:
        continue
    psnr = float(mean_row[2])
    ssim = float(mean_row[3])
    lpips = None if mean_row[4] in ('', 'None', None) else float(mean_row[4])
    rows.append((step, n, psnr, ssim, lpips, str(f)))

rows.sort(key=lambda x: x[0])
out = root / 'DermaRepo_ckpt_scan_summary.csv'
with open(out, 'w', newline='') as fp:
    w = csv.writer(fp)
    w.writerow(['step', 'num_samples', 'psnr', 'ssim', 'lpips', 'metrics_csv'])
    for r in rows:
        w.writerow(r)

print(f'wrote: {out}')
for r in rows:
    print(r)
