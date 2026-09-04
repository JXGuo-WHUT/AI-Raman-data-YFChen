"""
Generate a small SYNTHETIC Raman-like dataset so the repository is runnable
out of the box (no real / private spectra required).

The two classes differ only in a handful of Gaussian peaks, which makes them a
useful smoke test for the "--task all" feature-selection pipeline.

    python make_demo_data.py --n_per_class 40 --n_features 500
"""

import argparse

import numpy as np


def gaussian_peak(x, center, amplitude, width):
    return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=str, default='data/example.csv')
    p.add_argument('--n_per_class', type=int, default=40)
    p.add_argument('--n_features', type=int, default=500)
    p.add_argument('--begin', type=float, default=402.729,
                   help='Raman shift of the first feature.')
    p.add_argument('--step', type=float, default=1.6,
                   help='Spacing between two adjacent features (cm^-1).')
    p.add_argument('--seed', type=int, default=11)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    shifts = args.begin + args.step * np.arange(args.n_features)

    # 两个类别共有的背景峰 + 各自的特征峰
    shared = [(900, 3000), (1250, 2200), (1650, 1800)]
    markers = {
        0: [(1004, 2600), (1450, 1500)],   # 类别 0 的特征峰
        1: [(1085, 2400), (1580, 2000)],   # 类别 1 的特征峰
    }

    rows, labels = [], []
    for cls in (0, 1):
        for _ in range(args.n_per_class):
            spec = np.zeros_like(shifts)
            for c, a in shared:
                spec += gaussian_peak(shifts, c, a, 25.0)
            for c, a in markers[cls]:
                spec += gaussian_peak(shifts, c, a, 18.0)
            # 缓慢漂移的基线 + 噪声
            spec += 400 + 0.15 * (shifts - shifts[0])
            spec += rng.normal(0, 60.0, size=shifts.shape)
            rows.append(spec)
            labels.append(cls)

    x = np.vstack(rows)
    y = np.asarray(labels)
    order = rng.permutation(len(y))
    x, y = x[order], y[order]

    header = ','.join(f'{s:.3f}' for s in shifts) + ',class'
    body = '\n'.join(','.join(f'{v:.4f}' for v in row) + f',{int(lbl)}'
                     for row, lbl in zip(x, y))

    import os
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(header + '\n' + body + '\n')

    print(f'Synthetic dataset written to {args.output}: '
          f'{x.shape[0]} samples x {x.shape[1]} features, '
          f'classes {np.unique(y).tolist()}')


if __name__ == '__main__':
    main()
