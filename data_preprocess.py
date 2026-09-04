"""
Build a single merged CSV from per-class folders of raw spectrum CSV files.

Expected raw file layout (one CSV per spectrum, two columns:
Raman shift / intensity, optionally with a header line):

    raw/
      class_0/
        sample_001.csv
        sample_002.csv
      class_1/
        sample_003.csv

Each raw file contributes ONE row: the intensity column (from the row whose
first cell equals `--start_shift`) after airPLS baseline correction and
max-normalisation, plus the class index in the last column.

Example
-------
    python data_preprocess.py \
        --input_dirs raw/class_0 raw/class_1 \
        --output data/example.csv
"""

import argparse
import csv
import glob
import os

import numpy as np
from scipy import signal

from baseline import BASELINE_METHODS, correct_baseline


def parse_args():
    p = argparse.ArgumentParser(description='Merge raw spectrum CSVs into one dataset.')
    p.add_argument('--input_dirs', nargs='+', required=True,
                   help='One folder per class, in label order (folder 0 -> class 0, ...).')
    p.add_argument('--output', type=str, default='data/example.csv',
                   help='Path of the merged output CSV.')
    p.add_argument('--start_shift', type=str, default='402.729',
                   help='Raman shift value that marks the first usable data row.')
    p.add_argument('--encoding', type=str, default='cp1252',
                   help='Encoding of the raw CSV files.')
    p.add_argument('--baseline', type=str, default='airpls',
                   choices=list(BASELINE_METHODS),
                   help="Baseline correction method. 'airpls'/'als' need "
                        "pybaselines (BSD-3); 'detrend' uses scipy only; "
                        "'none' skips the step.")
    p.add_argument('--lambda_', type=float, default=1e5,
                   help='airPLS smoothness parameter (lam).')
    p.add_argument('--porder', type=int, default=3,
                   help='airPLS order of the difference penalties (diff_order).')
    p.add_argument('--itermax', type=int, default=50,
                   help='airPLS maximum number of iterations.')
    p.add_argument('--scale', type=float, default=10000.0,
                   help='Value that the max-normalised spectrum is scaled to.')
    p.add_argument('--quiet', action='store_true',
                   help='Suppress per-file warnings from the baseline fitter.')
    return p.parse_args()


# 平滑滤波算法
def SG(data, window=7):
    return signal.savgol_filter(data, window, 2)


# 归一化算法
def normalize(data, scale=10000.0):
    data = data / np.max(data)
    return data * scale


def main():
    opt = parse_args()

    all_data = []
    feature_names = None

    for class_idx, folder_path in enumerate(opt.input_dirs):
        csv_files = sorted(glob.glob(os.path.join(folder_path, '*.csv')))
        if not csv_files:
            print(f'[warn] no CSV found in {folder_path}')
            continue

        for file_path in csv_files:
            with open(file_path, 'r', encoding=opt.encoding) as f:
                rows = list(csv.reader(f))

            # 找到数据起始行（第一列等于起始拉曼位移的那一行）
            start = -1
            for idx, row in enumerate(rows):
                if row and row[0].strip() == opt.start_shift:
                    start = idx
                    break
            if start == -1:
                print(f'[warn] skip {file_path}: no row starting with {opt.start_shift}')
                continue

            if feature_names is None:
                feature_names = [row[0] for row in rows[start:] if row]
                feature_names.append('class')

            row_data = np.array([row[1] for row in rows[start:] if len(row) > 1], dtype=float)
            row_data = correct_baseline(
                row_data, method=opt.baseline, lam=opt.lambda_,
                diff_order=opt.porder, max_iter=opt.itermax, quiet=opt.quiet)
            row_data = normalize(row_data, opt.scale).tolist()
            row_data.append(class_idx)
            all_data.append(row_data)

    if not all_data:
        raise SystemExit('No spectrum was read — check --input_dirs / --start_shift.')

    os.makedirs(os.path.dirname(os.path.abspath(opt.output)), exist_ok=True)
    with open(opt.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(feature_names)
        writer.writerows(all_data)

    print(f'Saved {len(all_data)} samples x {len(feature_names) - 1} features to {opt.output}')


if __name__ == '__main__':
    main()
