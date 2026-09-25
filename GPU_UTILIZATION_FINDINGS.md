# GPU Utilization Diagnostic — QAMPARI Joint-Training Reproduction

## Background

The QAMPARI joint-training reproduction (`finetune_qampari_joint_repro.sh`,
2x H200, account-wide default accelerate config) was killed by the
cluster's automated policy: jobs under 60% GPU utilization for 2+ hours get
cancelled (75% is a warning threshold below that). This explains the
original `multi_hungarian_with_detach` checkpoint's history too (see
`experiment_plan.md`) — it was very likely killed by this same policy
repeatedly, not manually restarted.

**Root cause hypothesis:** the account-wide default accelerate config uses
FSDP with `fsdp_offload_params: true`. That constantly shuttles parameters
between CPU and GPU — designed for models too large for GPU memory, but
this is a 1.5B-parameter model (~3GB in fp16) on H200s (141GB each), so the
offload is unnecessary and its CPU↔GPU transfer overhead is a plausible
cause of low utilization.

## Test methodology

`gpu_util_test.sbatch`: runs the QAMPARI joint-training config
(`finetuning_multi.py`, same hyperparameters as the real reproduction, but
`--total_steps 100000 --not_save` so it never actually finishes or writes
checkpoints — this is a pure utilization diagnostic, not a real training
run) for a fixed 10-minute window (`timeout 600`), sampling
`nvidia-smi --query-gpu=utilization.gpu` across both GPUs every 5s. First
60s of samples discarded (model load / warmup). Reports mean/min/max
utilization over the remaining window.

## Results

| Test | Config | Batch/GPU | Mean util | Samples | Verdict |
|---|---|---|---|---|---|
| Fix 1 | FSDP, `fsdp_offload_params: false` | 50 (original) | **76.3%** | 214 | Clears both thresholds, thin margin over 75% warning line |
| Fix 1 + bigger batch | FSDP, `fsdp_offload_params: false` | 100 | ~~58.6%~~ **CRASHED (OOM)** | 114 (invalid, contaminated by crash) | `torch.OutOfMemoryError` at step 5 -- GPU memory hit 139.6/139.79GB. Without offload, doubling batch simply exceeds H200 VRAM. The 58.6% number is an artifact of the process dying partway through the 10-min window (util drops to ~0 after crash), not a real utilization measurement -- disregard it as a batch-size comparison point. |
| Fix 2 | Plain DDP (no FSDP) | 50 (original) | **77.3%** | 214 | Clean full-window run, no crash. Marginally better than Fix 1, and simpler (no FSDP sharding/communication overhead -- full model replica per GPU, which a 1.5B model comfortably affords) |

**Real QAMPARI reproduction training was launched using Fix 1** (job
`18270793`) before Test 2's result came back -- Fix 1 and Fix 2 perform
within noise of each other (76.3% vs 77.3%), so there's no strong reason to
kill and restart the already-running job over this difference. Fix 2 (DDP)
is recommended as the default for any *future* 2-GPU training in this repo
given its simplicity, but switching the current run isn't necessary.

## Recommendation

**Fix 1 (drop `fsdp_offload_params`) works** by the stated pass criterion
(above 60% cancel / 75% warning), and requires no change to training
hyperparameters — closest to a faithful reproduction of the original setup.

**Caveat:** 76.3% is only ~1 point above the 75% warning threshold, measured
over a 10-minute window. A real multi-hour run has more sources of
variability (periodic eval passes, data-loading hiccups, GPU scheduling
noise) that a 10-minute sample can't fully capture — there's real risk this
doesn't hold up over 2+ continuous hours. Given another cancellation costs
~2h of wasted 2-GPU time, I have not yet tested with more margin (e.g. DDP,
or a batch-size bump per the user's fallback plan) — holding for a decision
on whether 76.3% is good enough to commit to, or whether to test one more
config for a safer margin before resubmitting the real QAMPARI training job
(which remains un-submitted, per instruction).
