"""Aggregate per-run evaluation_summary.json into the summary CSVs (implementation.md 5.4-5.6).

Scans an experiments root for ``evaluation_summary.json`` files (written by
rectified_evaluate.py), parses variant/dataset/seed from the run directory name
(``<variant>_<ds>_seed<seed>``, e.g. ``gene2image_c1_seed42``), and produces
mean +/- std tables across seeds.

Metric keys map to the evaluator's actual schema:
    overall_fid, mean_batch_fid, mean_ssim, mean_psnr, overall_uni2h_fid.

Outputs:
    summary_main.csv      : per (variant, dataset) mean+/-std of each metric
    ablation/summary.csv   : same table tagged with the flipped switch / target RQ

Usage:
    python scripts/summarize_results.py --results_root results --out_dir results
"""
import os
import re
import json
import glob
import argparse
import numpy as np
import pandas as pd


RUN_RE = re.compile(r'(?P<variant>[a-zA-Z0-9]+)_(?P<ds>c1|c2|p1)_seed(?P<seed>\d+)')

SWITCH = {
    'gene2image': ('full', 'RQ1 main'),
    'geneflow':   ('no pathway encoder', 'lower bound'),
    'randpath':   ('real->random mask', 'RQ2 mechanism'),
    'pathprior':  ('learnable->frozen', 'RQ3 fixed scoring'),
    'notrans':    ('remove transformer', 'pathway co-regulation'),
    'nomask':     ('sparse->dense', 'structured sparsity'),
}

# eval json key -> friendly metric name (direction: fid/uni2h lower better, ssim/psnr higher)
METRIC_KEYS = {
    'overall_fid': 'fid',
    'mean_batch_fid': 'fid_batch',
    'mean_ssim': 'ssim',
    'mean_psnr': 'psnr',
    'overall_uni2h_fid': 'uni2h_fid',
}


def collect(results_root):
    """Return a long DataFrame of all evaluation_summary.json found under results_root."""
    rows = []
    pattern = os.path.join(results_root, '**', 'evaluation_summary.json')
    for jpath in glob.glob(pattern, recursive=True):
        # Identity comes from the run directory name (handles eval_on_* subdirs too:
        # walk up until a dir matches the run pattern).
        ident = None
        d = os.path.dirname(jpath)
        for _ in range(4):
            m = RUN_RE.search(os.path.basename(d))
            if m:
                ident = m
                break
            d = os.path.dirname(d)
        with open(jpath) as f:
            data = json.load(f)
        rec = {}
        if ident:
            rec['variant'] = ident.group('variant')
            rec['dataset'] = ident.group('ds')
            rec['seed'] = int(ident.group('seed'))
        else:
            rec['variant'] = data.get('encoder_type', 'unknown')
            rec['dataset'] = 'unknown'
            rec['seed'] = data.get('seed')
        rec['eval_path'] = jpath
        for src_key, name in METRIC_KEYS.items():
            v = data.get(src_key)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                rec[name] = v
        rows.append(rec)
    return pd.DataFrame(rows)


def aggregate(df):
    """Mean +/- std across seeds for each (variant, dataset)."""
    if df.empty:
        return df
    present = [m for m in METRIC_KEYS.values() if m in df.columns]
    g = df.groupby(['variant', 'dataset'])[present].agg(['mean', 'std'])
    g.columns = [f'{m}_{stat}' for m, stat in g.columns]
    return g.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_root', default='results')
    ap.add_argument('--out_dir', default='results')
    args = ap.parse_args()

    df = collect(args.results_root)
    if df.empty:
        print(f"No evaluation_summary.json found under {args.results_root}. "
              f"Run experiments + evaluation first.")
        return
    print(f"Collected {len(df)} eval runs.")
    summary = aggregate(df)
    os.makedirs(args.out_dir, exist_ok=True)

    main_path = os.path.join(args.out_dir, 'summary_main.csv')
    summary.to_csv(main_path, index=False)
    print(f"wrote {main_path} ({len(summary)} variant x dataset rows)")

    abl = summary.copy()
    abl['flipped_switch'] = abl['variant'].map(lambda v: SWITCH.get(v, ('', ''))[0])
    abl['target_rq'] = abl['variant'].map(lambda v: SWITCH.get(v, ('', ''))[1])
    abl_dir = os.path.join(args.out_dir, 'ablation')
    os.makedirs(abl_dir, exist_ok=True)
    abl_path = os.path.join(abl_dir, 'summary.csv')
    abl.to_csv(abl_path, index=False)
    print(f"wrote {abl_path}")


if __name__ == "__main__":
    main()
