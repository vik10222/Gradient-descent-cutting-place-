# -------------------------------------------------------------
#  Self‑contained AdamW hyper‑parameter grid search script
#  (minimal reproducible example using your exact problem setup)
# -------------------------------------------------------------
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

# ─── 1.  Problem‑specific helpers (unchanged from your code) ────

def _sign_subgrad(z, tol=1e-12):
    g = np.zeros_like(z)
    g[z >  tol] =  1.0
    g[z < -tol] = -1.0
    return g


def l1_regression(p, A_square, b_obs, lam):
    r = A_square @ p - b_obs
    return 0.5 * r.dot(r) + lam * np.abs(p).sum()


def subgrad_l1_regression(p, A_square, b_obs, lam):
    r = A_square @ p - b_obs
    return A_square.T @ r + lam * _sign_subgrad(p)


# ─── 2.  AdamW implementation (verbatim logic) ─────────────────

def adamW(
    initial_point,
    f,
    grad,
    lr=1e-1,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    weight_decay=1e-2,
    num_steps=10_000,
):
    """ Return best parameters and best f(x) found during run. """
    x = initial_point.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)

    warm_up   = 40
    const_up  = 2_000
    window    = 30
    patience  = 2
    delta     = 1e-4

    lr_t_min  = lr * 1e-2      # minimal LR after decay
    factor    = 0.5            # LR decay factor when plateaued
    bad_win   = 0

    f_hist = [f(x)]
    best_val = f_hist[0]

    for t in range(1, num_steps + 1):
        # simple 1‑cycle style LR schedule w/ plateau detection
        if t <= warm_up:
            lr_t = lr * t / warm_up
        elif t <= const_up:
            lr_t = lr
        else:
            lr_t = max(lr / np.sqrt(t), lr_t_min)

        g = grad(x)

        # 1) decoupled weight decay
        x *= 1 - lr_t * weight_decay

        # 2) moment estimates
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g * g)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        # 3) parameter update
        x -= lr_t * m_hat / (np.sqrt(v_hat) + eps)

        # 4) log + best‑val tracking
        fx = f(x)
        f_hist.append(fx)
        if fx < best_val:
            best_val = fx

        # 5) crude plateau logic every step after const_up
        if t > const_up and len(f_hist) > window:
            if (f_hist[-window-1] - f_hist[-1]) / (abs(f_hist[-window-1]) + 1e-12) < delta:
                bad_win += 1
                if bad_win >= patience:
                    lr = max(lr * factor, lr_t_min)  # decay base LR
                    bad_win = 0
            else:
                bad_win = 0

    return x, best_val


# ─── 3.  Grid‑search utility  ──────────────────────────────────

def grid_search_adam(x0, f, grad, param_grid, num_steps=10_000):
    records = []
    for cfg in ParameterGrid(param_grid):
        t0 = time.perf_counter()
        _, best_val = adamW(x0, f, grad, num_steps=num_steps, **cfg)
        elapsed = time.perf_counter() - t0
        records.append({**cfg, "best_val": best_val, "elapsed_s": elapsed})
    return pd.DataFrame(records).sort_values("best_val").reset_index(drop=True)


# ─── 4.  Main: build test instance & run search  ───────────────

def main():
    # instance identical to your original script
    d          = 20
    A_square   = np.vstack([np.eye(d), -np.eye(d)])
    b_obs      = np.zeros(2 * d)
    lam        = 0.1

    f_closure    = lambda x: l1_regression(x, A_square, b_obs, lam)
    grad_closure = lambda x: subgrad_l1_regression(x, A_square, b_obs, lam)

    rng  = np.random.default_rng(42)
    x0   = rng.uniform(-40, 40, size=d)

    # hyper‑param grid (feel free to expand!)
    param_grid = {
        "lr":           [1e-1, 5e-2, 1e-2, 1e-3],
        "beta1":        [0.9, 0.95],
        "beta2":        [0.999, 0.9999],
        "weight_decay": [1e-2, 1e-3],
    }

    df = grid_search_adam(x0, f_closure, grad_closure, param_grid)
    pd.set_option("display.float_format", lambda v: f"{v:.3e}")
    print("\n=== AdamW grid search results (best → worst) ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
