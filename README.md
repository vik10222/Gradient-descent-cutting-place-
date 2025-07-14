# GD‑Cutting Plane + AdamW Hybrid Optimizer

A multiprocessing‑friendly implementation of a gradient‑descent cutting‑plane method (“GD‑cutting2”) that explores and refines a feasible region via linear cuts, then hands off to AdamW for final convergence. Designed for `l1`‑regularized least‑squares problems but extensible to any differentiable objective with simple linear constraints. Note. cutting plane aspect should be turned of for non convex problem


---

## Features

- **Shared cuts pool**: Workers collaboratively accumulate global cutting planes.
- **Feasibility projection**: Infeasible iterates are projected onto the most violated hyperplane.
- **Pruning strategies**: Drop redundant cuts via simple “LP” thresholding or a slack‑based filter.
- **Adaptive line search**: Local step‑size growth/decay within each trial.
- **Phase switching**: Automatically move from “explore” (cut‐accumulation) to “converge” (AdamW) based on recent improvement.
- **AdamW backend**: After cutting‐plane phase, continue with a bias‑corrected, decoupled‑weight‑decay Adam optimizer.
- **Anytime curves**: Track best objective vs. iteration count and wall‐clock time, per worker and globally.

---

## Requirements

- Python 3.8+
- NumPy
- pandas
- matplotlib
- scikit‑learn
- (Optionally) `multiprocessing` for parallel runs

---

## Installation

```bash
git clone https://github.com/yourname/gd-cutting-adamw.git
cd gd-cutting-adamw
pip install -r requirements.txt
