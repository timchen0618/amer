"""The LR schedule must advance exactly once per training step, on any number of processes.

Builds the real optimizer/scheduler (utils.set_optim) and the real training Accelerator
(finetuning_multi.build_accelerator) around a tiny model, runs `total_steps` optimizer +
scheduler steps, and checks every LR against WarmupLinearScheduler's formula. A negative
control repeats the run with accelerate's default (step_scheduler_with_optimizer=True),
which advances the schedule num_processes times per step, to show the check catches it.

Run from the repo root, e.g.:
    accelerate launch --config_file training/inf_retriever/accelerate_config_2gpu_ddp.yaml \
        training/inf_retriever/tests/test_scheduler_stepping.py
    accelerate launch --config_file training/inf_retriever/accelerate_config_1gpu.yaml \
        training/inf_retriever/tests/test_scheduler_stepping.py
"""
import argparse
import os
import sys

import torch
from accelerate import Accelerator

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src import utils  # noqa: E402
from finetuning_multi import build_accelerator  # noqa: E402


def make_opt(total_steps, warmup_steps, lr):
    return argparse.Namespace(
        optim="adamw", scheduler="linear", lr=lr, beta1=0.9, beta2=0.98, eps=1e-6,
        weight_decay=0.01, warmup_steps=warmup_steps, total_steps=total_steps,
        lr_min_ratio=0.0, accumulation_steps=1,
    )


def expected_lr(t, opt):
    """WarmupLinearScheduler after t scheduler steps (ratio=0)."""
    if t < opt.warmup_steps:
        return opt.lr * t / max(1, opt.warmup_steps)
    return opt.lr * max(0.0, 1.0 - (t - opt.warmup_steps) / max(1.0, opt.total_steps - opt.warmup_steps))


def run(accelerator, opt):
    model = torch.nn.Linear(4, 1)
    optimizer, scheduler = utils.set_optim(opt, model)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    x = torch.ones(2, 4, device=accelerator.device)
    lrs = []
    for _ in range(opt.total_steps):
        loss = model(x).sum()
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        lrs.append(scheduler.get_last_lr()[0])
    return lrs, scheduler.scheduler.last_epoch


def main():
    opt = make_opt(total_steps=2500, warmup_steps=100, lr=1e-5)

    # 1) Production accelerator: one scheduler step per training step.
    accelerator = build_accelerator(opt, log_with=None)
    nproc = accelerator.num_processes
    lrs, last_epoch = run(accelerator, opt)
    errors = [(t, lr, expected_lr(t, opt)) for t, lr in enumerate(lrs, start=1)
              if abs(lr - expected_lr(t, opt)) > 1e-12]
    checks = {
        "every step matches the formula": not errors,
        "scheduler steps == training steps": last_epoch == opt.total_steps,
        "lr at step 50 == 5e-6 (half of warmup 100)": abs(lrs[49] - 5e-6) < 1e-12,
        "lr at step 100 == 1e-5 (end of warmup)": abs(lrs[99] - 1e-5) < 1e-12,
        "lr > 0 for every step before the last": all(lr > 0 for lr in lrs[:-1]),
        "lr == 0 only after the final step": lrs[-1] == 0.0,
    }

    # 2) Negative control: accelerate's default steps the schedule num_processes times.
    ctrl = Accelerator(mixed_precision="bf16")  # step_scheduler_with_optimizer defaults to True
    ctrl_lrs, ctrl_last_epoch = run(ctrl, opt)
    first_zero = next((t for t, lr in enumerate(ctrl_lrs, start=1) if lr == 0.0), None)

    if accelerator.is_main_process:
        print(f"num_processes={nproc}")
        for name, ok in checks.items():
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if errors:
            print(f"  first mismatches (step, got, expected): {errors[:5]}")
        print(f"  lr @ steps 1/50/100/1300/2499/2500: "
              f"{[f'{lrs[i-1]:.3g}' for i in (1, 50, 100, 1300, 2499, 2500)]}")
        print(f"negative control (accelerate default): scheduler steps={ctrl_last_epoch} "
              f"for {opt.total_steps} training steps; lr first hits 0 at step {first_zero}")
        if nproc > 1:
            ctrl_ok = ctrl_last_epoch == nproc * opt.total_steps or first_zero == opt.total_steps // nproc
            print(f"  [{'PASS' if ctrl_ok else 'FAIL'}] control reproduces the {nproc}x double-step bug")
        ok = all(checks.values())
        print("RESULT:", "PASS" if ok else "FAIL")
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
