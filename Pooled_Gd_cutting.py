import numpy as np
import multiprocessing as mp
from multiprocessing import current_process
import time
import argparse
import pandas as pd

from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt


#this si basically the code I ran for the parametergrid search, but found some error so this is really it
#this is the shit basically




# ─── 2. SHARED FUNCTION DEFINITIONS ─────────────────────────────
def _sign_subgrad(z, tol=1e-12):
    g = np.zeros_like(z)
    g[z >  tol] =  1.0
    g[z < -tol] = -1.0
    return g

def l1_regression(p, A_square, b_obs, λ):
    r = A_square @ p - b_obs
    return 0.5 * r.dot(r) + λ * np.abs(p).sum()

def subgrad_l1_regression(p, A_square, b_obs, λ):
    r = A_square @ p - b_obs
    return A_square.T @ r + λ * _sign_subgrad(p)

def is_feasible(pt, A0, b0):
    return np.all(A0 @ pt <= b0)

def project_to_nearest_face(pt, A0, b0):
    v   = A0 @ pt - b0
    idx = np.argmax(v)
    a,b = A0[idx], b0[idx]
    t   = (b - a.dot(pt)) / (a.dot(a) + 1e-12)
    return pt + t*a, idx

def prune_lp(A0, b0, eps=1e-8):
    keep = np.abs(A0).sum(axis=1) > eps
    return A0[keep], b0[keep]

def update_radius(r_prev, new_pt, ref_pt, radius_smooth):
    """Exponential moving max of ‖new_pt − ref_pt‖."""
    return max(radius_smooth * r_prev,
               np.linalg.norm(new_pt - ref_pt))


def GD_cutting2(initial_point, line_search_steps, data, optimum_val, step_size, step_growth, step_decay, path=None ,num_trials=10, prune_at=1000, prune_strat="LP", Gamma_slack=1.5, phase =None, radius_smooth=0.95, base_time= None):

    """
    the differrence here is that when we are infeasible we project that point onto the feasible and gettign the gradient there and continue
    """

    pt = initial_point.copy()
    best = (initial_point.copy(),data['f'](initial_point))
    curr_step = step_size

   
    path = [initial_point]
    trials =0
    window =100
    min_f_improvement=0.02*len(initial_point)

    search_radius = np.linalg.norm(best[0] - initial_point) + 1e-12



    f_history= [best[1]]
    t_hist  = [0.0]
    if base_time==None:
        base_time = time.perf_counter()
    while trials<num_trials and phase !="converge":  


        if len(f_history) > window and phase=="explore":
                # take the last window+1 values
                recent = f_history[-(window+1):]
                # compute the per-step drops
                drops = [recent[i] - recent[i+1] for i in range(window)]
                if sum(drops) < window * min_f_improvement:
                    phase = "converge"
                    break
 

    
        counter= 0
        grad = data['grad'](best[0])
        pt = best[0].copy()

        a = grad
        b = float(np.dot(grad, best[0]))

            # 1) push local cut into the global pool & grab the full list
        with CUT_LOCK:
            CUTS.append((a.tolist(), b))
            all_cuts = list(CUTS)   # snapshot everything now in CUTS

        # 2) rebuild your local arrays from that global snapshot
        data["A0"] = np.vstack([c[0] for c in all_cuts])
        data["b0"] = np.array([c[1] for c in all_cuts])

        if prune_strat=="LP"  and data["A0"].shape[0] > prune_at:
            data["A0"], data["b0"] =prune_lp(data["A0"], data["b0"], eps=1e-8)
        elif prune_strat=="Slack" and data["A0"].shape[0] > prune_at:
            slacks = data["b0"] - data["A0"] @ best[0]
            keep   = slacks <= Gamma_slack * search_radius
            data["A0"], data["b0"] = data["A0"][keep], data["b0"][keep]
                    

        while counter < line_search_steps:
            if is_feasible(pt, data["A0"], data["b0"]) == False:
                pt, ai= project_to_nearest_face(pt, data["A0"], data["b0"])
                grad =data['grad'](pt)
                curr_step = max(curr_step * step_decay, 1e-2)
                counter +=1
                continue

            else:
                temp_f=data['f'](pt)
                f_history.append(temp_f)
                t_hist.append(time.perf_counter() - base_time)
                if temp_f< best[1]:
                    best = (pt.copy(), temp_f)

            curr_step = curr_step*step_growth
            pt = pt - curr_step*grad
            counter +=1
        trials +=1
        path.append(best[0])

        search_radius = update_radius(search_radius, best[0], initial_point, radius_smooth)





    print(f"--------------------------Processor: {current_process().name[-1]} trial: {trials} best value: {best[1]}--------------")

    if phase == "converge" and trials<num_trials:
        print("ADAM TIME")
        best_a, best_a_val, _,  adam_hist, adam_t_hist =adamW(best[0], 
            data["f"],         # f(x) → scalar, already knows A_square, b_obs, λ
            data["grad"],
            lr=1e-3,          # base learning rate
            beta1=0.9, 
            beta2=0.999, 
            eps=1e-8, 
            weight_decay=1e-2,# decoupled weight decay coeff
            num_steps=500, path=path, start_time=base_time)
        
        # print(f"adam best: {best_a_val}")
        # print(f"gd_c best:  {best[1]}")
        if best_a_val<best[1]:
            best = (best_a, best_a_val)
     
        f_history.extend(adam_hist[1:])
        t_hist.extend(adam_t_hist)
    best_pt, best_val = best
    return best_pt, best_val, f_history, t_hist
  

    # ─── 3. shared-cuts infrastructure ──────────────────────────────
def init_worker(shared_cuts, shared_lock):
    global CUTS, CUT_LOCK
    CUTS, CUT_LOCK = shared_cuts, shared_lock

def run_one(initial_point, line_steps,
            A_square, b_obs, lam,          # ← 3 new items
            optimum_val, ss, sg, sd,
            num_trials, prune_at,prune_strat, Gamma_slack, phase,radius_smooth, base_time):
    

    # build picklable callables inside the worker
    f    = lambda x: l1_regression(x, A_square, b_obs, lam)
    grad = lambda x: subgrad_l1_regression(x, A_square, b_obs, lam)

    path = None

    with CUT_LOCK:
        A0 = np.array([c[0] for c in CUTS])
        b0 = np.array([c[1] for c in CUTS])

    data = {"f": f, "grad": grad, "A0": A0, "b0": b0}

    best_pt, best_val, f_hist, t_hist = GD_cutting2(
        initial_point, line_steps, data,
        optimum_val, ss, sg, sd,
        path, num_trials, prune_at, prune_strat, Gamma_slack, phase,radius_smooth, base_time)

    return best_pt, best_val, path, f_hist, t_hist


def run_adam_worker(start_pt,
                    A_square, b_obs, lam,
                    lr=1e-3, beta1=0.9, beta2=0.999,
                    eps=1e-8, weight_decay=1e-2,
                    num_steps=5_000):
    f    = lambda x: l1_regression(x, A_square, b_obs, lam)
    grad = lambda x: subgrad_l1_regression(x, A_square, b_obs, lam)
    x, fx, path, f_hist, t_hist = adamW(start_pt, f, grad,
                                        lr, beta1, beta2,
                                        eps, weight_decay,
                                        num_steps)
    return x, fx, path, f_hist, t_hist




def adamW(initial_point, 
          f,                # objective: f(x)
          grad,             # gradient: ∇f(x)
          lr=1e-1,          # base learning rate
          beta1=0.9, 
          beta2=0.999, 
          eps=1e-8, 
          weight_decay=1e-2,# decoupled weight decay coeff
          num_steps=100, path=None, start_time=None, ):
    """
    Simple AdamW optimizer.
    Args:
      initial_point: np.ndarray, starting parameters
      f: callable, f(x) → scalar (unused here but helpful for logging)
      grad: callable, ∇f(x) → np.ndarray
      lr, beta1, beta2, eps: Adam hyperparams
      weight_decay: λ_W in decoupled WD
      num_steps: total update steps
    Returns:
      x: np.ndarray, final parameters
    """
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




    

    f_hist=[f(initial_point)]
    if start_time ==None:
        t_hist = [0.0]
        start_time=time.perf_counter()
    else:
        t_hist = [start_time]


    if path == None:
       path= [initial_point.copy()]

    for t in range(1, num_steps + 1):

        if t <= warm_up:
            lr_t = lr * t / warm_up
        elif t <= const_up:
            lr_t = lr
        else:
           lr_t = max(lr / np.sqrt(t), lr_t_min)

       
       
        g = grad(x)

        # 1) decoupled weight decay step
        x *= (1 - lr * weight_decay)

        # 2) moment updates
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g * g)

        # 3) bias correction
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        # if t ==1:
        #     v_hat = v / (1 - beta2**t)
        # else:
        #     v_hat=np.maximum((v / (1 - beta2**t)),v_hat)

        # 4) parameter update
        x -= lr_t * m_hat / (np.sqrt(v_hat) + eps)
        path.append(x)
        f_hist.append(f(x))
        t_hist.append(time.perf_counter() - start_time)


        if t > const_up and len(f_hist) > window:
            if (f_hist[-window-1] - f_hist[-1]) / (abs(f_hist[-window-1]) + 1e-12) < delta:
                bad_win += 1
                if bad_win >= patience:
                    lr = max(lr * factor, lr_t_min)  # decay base LR
                    bad_win = 0
            else:
                bad_win = 0

    return x, f(x), path,f_hist, t_hist


# ─── 3. MAIN ENTRYPOINT ─────────────────────────────────────────
def main():
    results = []
    total = 1000
    # unpack params
    d              = 20
    line_steps     = 100
    ss             = 0.25
    sg             = 1.1
    sd             = 0.1
    num_trials     = 100
    prune_at       = 50
    num_proc = 4
    prune_strat    = 'Slack'
    Gamma_slack    = 0.75
    phase          = 'explore'      # 'explore'   None
    radius_smooth  = 0.95

    # build problem data
    A_square = np.vstack([np.eye(d), -np.eye(d)])
    b_box    = np.full(2*d, 5.0)
    b_obs    = np.zeros(2*d)
    λ         = 0.1
    optimum_val = l1_regression(np.zeros(d), A_square, b_obs, λ)
    f_closure    = lambda x: l1_regression(x, A_square, b_obs, λ)
    grad_closure = lambda x: subgrad_l1_regression(x, A_square, b_obs, λ)



    # shared cuts manager
    mgr  = mp.Manager()
    cuts = mgr.list()
    lock = mgr.Lock()
    for a,b in zip(A_square, b_box):
        cuts.append((a.tolist(), float(b)))

    # initial points
    seed = 42                        # pick any integer you like
    rng  = np.random.default_rng(seed)
    init_pts = rng.uniform(-40, 40, size=(num_proc, d)) #could seed this

    adam_results ={"adam_best_pt":}
    t0_adam=time.time()
    for pts in init_pts:
        adam_best_pt, adam_best_val, adam_path, adam_f_hist, adam_t_hist= adamW(pts, 
            f_closure, grad_closure,
            lr=1e-1,          # base learning rate
            beta1=0.9, 
            beta2=0.999, 
            eps=1e-8, 
            weight_decay=1e-2,# decoupled weight decay coeff
            num_steps=10000)
        t1_adaM = time.time()
        elapse_adam = t1_adaM - t0_adam
    

    # spawn pool & run
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=num_proc,
                    initializer=init_worker,
                    initargs=(cuts, lock)) as pool:
        

        base_time = time.perf_counter()

        batch = [
            (pt, line_steps,
                A_square, b_obs, λ,            # ⚑ raw arrays, no lambda
                optimum_val, ss, sg, sd,
                num_trials, prune_at, prune_strat, Gamma_slack, phase, radius_smooth, base_time)
            for pt in init_pts
        ]


        t0 = time.time()
        results_batch = pool.starmap(run_one, batch)
        elapsed = time.time() - t0
        gdc_hists = [r[3] for r in results_batch] 
        gdc_t_hists = [r[4] for r in results_batch]

    # post-process and print one grid‐search resultf
    best_pt, best_val, _, _ , _= min(results_batch, key=lambda r: r[1])

    fmt = lambda x: f"{x:.3e}"
    print(f"[d={d},ss={ss},sg={sg},sd={sd},tr={num_trials},pr={prune_at}]",
            f"best_val={fmt(best_val)}, time={fmt(elapsed)}s, best adaam val ={fmt(adam_best_val)} adam time={fmt(elapse_adam)}")
    


    # store
    max_len = max(len(h) for h in gdc_hists)

    gdc_padded = [
        np.pad(h, (0, max_len - len(h)), constant_values=h[-1])
        for h in gdc_hists
    ]                                   # shape = (#workers, max_len)

    # per-worker anytime curve
    gdc_anytime_worker = [np.minimum.accumulate(h) for h in gdc_padded]

    # “best of all workers so far”
    gdc_anytime_global = np.minimum.accumulate(
        np.min(np.vstack(gdc_anytime_worker), axis=0)
    )                                   # shape = (max_len,)

    # --- AdamW (single run) ------------------------------------------
    adam_anytime = np.minimum.accumulate(adam_f_hist)

    
    x_gdc   = np.arange(len(gdc_anytime_global))      # 0‥999
    x_adam  = np.arange(len(adam_anytime))            # 0‥9999

    fig, ax_gdc = plt.subplots(figsize=(7,4))

    # ── bottom axis: GD-cutting ──────────────────────────────
    ax_gdc.plot(x_gdc, gdc_anytime_global, label='GD-cutting (global best)')
    for h in gdc_anytime_worker:                       # faint per-worker lines
        ax_gdc.plot(x_gdc[:len(h)], h, alpha=0.2, lw=0.8)

    ax_gdc.set_xlabel('GD-cutting iterations')
    ax_gdc.set_yscale('log')
    ax_gdc.set_ylabel('Best f(x) so far')

    # ── top axis: AdamW ──────────────────────────────────────
    ax_adam = ax_gdc.twiny()              # second x-axis that shares the y-axis
    ax_adam.plot(x_adam, adam_anytime, color='C1', label='AdamW')
    ax_adam.set_xlabel('AdamW iterations')

    # ── make the two x-axes line up end-to-end ───────────────
    # map GD range [0, len(gdc)-1] onto Adam range [0, len(adam)-1]
    scale = (len(x_adam)-1)/(len(x_gdc)-1)
    ax_adam.set_xlim(ax_gdc.get_xlim()[0]*scale, ax_gdc.get_xlim()[1]*scale)

    # ── one combined legend ─────────────────────────────────
    handles1, labels1 = ax_gdc.get_legend_handles_labels()
    handles2, labels2 = ax_adam.get_legend_handles_labels()
    ax_gdc.legend(handles1+handles2, labels1+labels2, loc='upper right')

    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # ❶ PLOT raw f(x) traces for every worker + Adam
    # ------------------------------------------------------------
    fig_raw, ax_raw = plt.subplots(figsize=(7,4))

    # --- GD-cutting workers -------------------------------------
    for i, hist in enumerate(gdc_hists, 1):
        ax_raw.plot(hist, label=f'Worker {i}', lw=1.0, alpha=0.8)


    ax_raw.set_yscale('log')                # keep vertical scale consistent
    ax_raw.set_xlabel('Function evaluations (per algorithm)')
    ax_raw.set_ylabel('Raw f(x)')
    ax_raw.set_title('Objective value vs. evaluation')
    ax_raw.legend(ncol=2, fontsize=9)
    fig_raw.tight_layout()
    plt.show()

    max_len = max(
    max(len(h) for h in gdc_hists),
    max(len(t) for t in gdc_t_hists))
    f_pad = [np.pad(f, (0, max_len - len(f)), constant_values=f[-1])
         for f in gdc_hists]

    t_pad = [np.pad(t, (0, max_len - len(t)), constant_values=t[-1])
         for t in gdc_t_hists]


    f_pad = [np.pad(f, (0,max_len-len(f)), constant_values=f[-1]) for f in gdc_hists]
    t_pad = [np.pad(t, (0,max_len-len(t)), constant_values=t[-1]) for t in gdc_t_hists]

    best_global = np.minimum.accumulate(np.min(np.vstack(f_pad), axis=0))
    time_global = np.min(np.vstack(t_pad), axis=0)          # earliest eval time

    fig_any, ax_any = plt.subplots(figsize=(7,4))

    ax_any.plot(time_global, best_global,
                label='GD-cutting (best across workers)')
    ax_any.plot(adam_t_hist,
                np.minimum.accumulate(adam_f_hist),
                label='AdamW')

    ax_any.set_xlabel('Wall-clock seconds')
    ax_any.set_ylabel('Best objective so far')
    ax_any.set_yscale('log')
    ax_any.legend()
    fig_any.tight_layout()
    plt.show()


if __name__ == "__main__":

    main()
