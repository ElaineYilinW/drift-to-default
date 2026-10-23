"""Linear chunk-recursion model of relaxation (exploratory theory).

A chunk's mean statistic x_k (e.g. L*) is modelled as
    x_{k+1} - mu = w_w * (mean of the window chunks - mu) + w_s * (x_sink - mu) [+ noise]
where the window holds the last `win` chunks (the held opening counts as a window chunk until it rolls out; after
that, with a sink, it keeps a direct pull w_s as measured by KV-cache replay), mu is the prompt's default and w_p = 1 - w_w - w_s is the pull toward it.
Persistence alpha(k) = d x_k / d x_0. With a sink the equilibrium share is alpha_inf = w_s / (1 - w_w); without one
alpha -> 0. Chunks are 3 latent frames = 12 video frames = 0.75 s at 16 fps.

usage (library): alpha_curve(w_w, w_s, win, sink, n_chunks); fit_curve(t_seconds, alpha, win, sink)
"""
import numpy as np
from scipy.optimize import least_squares

CHUNK_S = 0.75


def alpha_curve(w_w, w_s, win, sink, n_chunks):
    """d x_k / d x_0 for k = 0..n_chunks-1 (x_0 = held opening chunk)."""
    a = [1.0]
    for k in range(1, n_chunks):
        lo = max(0, k - win)                         # window = last `win` chunks before k (the opening included while
        w = np.mean(a[lo:k])                         # it is still among them)
        a.append(w_w * w + (w_s * a[0] if sink and lo > 0 else 0.0))   # sink pull once the opening left the window
    return np.array(a)


def fit_curve(t, alpha, win, sink, w_s=None):
    """Least-squares fit of (w_w, w_s) (or w_w only, given w_s) to an observed persistence curve alpha(t)."""
    t = np.asarray(t)
    k = np.clip(np.round(t / CHUNK_S).astype(int), 0, None)
    n = int(k.max()) + 1

    def resid(p):
        ww, ws = (p[0], p[1]) if w_s is None else (p[0], w_s)
        return alpha_curve(ww, ws, win, sink, n)[k] - alpha

    p0 = [0.85, 0.04] if w_s is None else [0.85]
    lb = [0.0, 0.0] if w_s is None else [0.0]
    ub = [0.999, 0.5] if w_s is None else [0.999]
    r = least_squares(resid, p0, bounds=(lb, ub))
    ww, ws = (r.x[0], r.x[1]) if w_s is None else (r.x[0], w_s)
    return dict(w_w=float(ww), w_s=float(ws), w_p=float(1 - ww - ws),
                alpha_inf=float(ws / (1 - ww)) if sink else 0.0, rmse=float(np.sqrt(np.mean(r.fun ** 2))))


if __name__ == "__main__":
    for ww in (0.76, 0.85, 0.90):
        for sink in (True, False):
            a = alpha_curve(ww, 0.036, 6 if sink else 7, sink, 42)
            print(f"w_w={ww} sink={sink}: alpha at 5/10/20/30 s =",
                  [round(float(a[int(s / CHUNK_S)]), 3) for s in (5, 10, 20, 30)],
                  "alpha_inf", round(0.036 / (1 - ww), 3) if sink else 0)
