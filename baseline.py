"""
Baseline correction for spectra.

We deliberately do NOT vendor a copy of the classic Renato-Lombardo airPLS
Python port, because that file is distributed under the LGPL-3.0. Instead we
call `pybaselines` (BSD-3-Clause, Copyright (c) 2021 Donald Erb), which ships
a well-tested airPLS implementation.

    pip install pybaselines

If `pybaselines` is not installed we fall back to a simple `scipy.signal`
detrend (BSD-3-Clause), or `--baseline none` to skip the step entirely.
"""

import warnings

import numpy as np

__all__ = ['correct_baseline', 'BASELINE_METHODS']

BASELINE_METHODS = ('airpls', 'als', 'detrend', 'none')


def _with_pybaselines(y, method, lam, diff_order, max_iter, tol):
    from pybaselines import whittaker

    if method == 'als':                       # asymmetric least squares
        baseline, _params = whittaker.asls(
            y, lam=lam, diff_order=diff_order, max_iter=max_iter, tol=tol)
    else:                                     # airPLS (default)
        baseline, _params = whittaker.airpls(
            y, lam=lam, diff_order=diff_order, max_iter=max_iter, tol=tol)
    return y - baseline


def _detrend(y):
    from scipy import signal
    return signal.detrend(np.asarray(y, dtype=float))


def correct_baseline(y, method='airpls', lam=1e5, diff_order=3,
                     max_iter=50, tol=1e-3, quiet=False):
    """Remove the slowly varying background from a single spectrum.

    Parameters
    ----------
    y : array-like, shape (n_features,)
    method : {'airpls', 'als', 'detrend', 'none'}
        'airpls' / 'als' require `pybaselines` (BSD-3-Clause).
    lam, diff_order, max_iter, tol : airPLS hyper-parameters
        `diff_order` is the order of the difference penalty
        (called `porder` in the original airPLS code).
    quiet : if True, suppress convergence warnings from pybaselines.

    Returns
    -------
    np.ndarray of the same shape as `y`.
    """
    y = np.asarray(y, dtype=float).ravel()

    if method == 'none':
        return y
    if method == 'detrend':
        return _detrend(y)
    if method not in ('airpls', 'als'):
        raise ValueError(f'Unknown baseline method: {method} '
                         f'(choose from {BASELINE_METHODS})')

    try:
        if quiet:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                return _with_pybaselines(y, method, lam, diff_order, max_iter, tol)
        return _with_pybaselines(y, method, lam, diff_order, max_iter, tol)
    except ImportError as exc:
        raise ImportError(
            'Baseline correction with airPLS/ALS requires pybaselines:\n'
            '    pip install pybaselines\n'
            'Alternatively pass --baseline detrend (uses scipy only) '
            'or --baseline none to skip this step.'
        ) from exc
