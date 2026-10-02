# 24-run grid results

*Status: placeholder, results to be filled in.*

## Code

| Commit | Contents |
| --- | --- |
| `4298660` | Training and data code the grid ran on: clean splits (`data_creation/build_clean_splits.py`), hard-negative miner, training driver (`phaseC_run.sh`, `phaseC_train.sbatch`, `grid_launch.sh`) |
| `c9e1b9e` | Eval pipeline used for the full-corpus evaluations (`retrieval_inf.py`, `full_eval_queue.sh`, embed / retrieve sbatch scripts) |

Branch `fsdp-clean-recipe`. Grid launched 2026-09-28; launch-time snapshot of the uncommitted diff: `results/phaseC/grid_code_snapshot.patch`.

Known issues that affect how these results read: `code_audit_2026-09-28.md` (in particular G1, G2, L1, C2).

## Grid

dataset {QAMPARI, AmbigQA} × mode {single, multi} × data {no hard neg., hard neg.} × LR {1e-6, 3e-6, 1e-5}.

## Results

### QAMPARI

### AmbigQA

## Notes
