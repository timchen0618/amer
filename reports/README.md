# reports/

Newest first. **Current** = the reference for its topic; the rest are kept as history.

| Report | Status | What it is |
|---|---|---|
| `objective_audit_2026-10-13/N_qwen3_final_report_2026-11-19.md` | Current | Session N: Qwen3-Embedding-0.6B replication and the legacy one-label step |
| `repo_organization_proposal_2026-11-18.md` | Superseded | First repo-cleanup proposal. What was actually done: this README, `results/README.md`, `training/inf_retriever/tools/README.md`, 25 root scripts to `archive/` |
| `objective_audit_2026-10-13/N_nemotron_final_report_2026-11-15.md` | Current | Session N: Nemotron retriever replication (final) |
| `objective_audit_2026-10-13/N_nemotron_results_2026-11-13.md` | History | Session N: Nemotron round-1 notes |
| `objective_audit_2026-10-13/F_retriever_swap_handoff_2026-11-12.md` | History | Session F's handoff to N for the retriever swap |
| `paper_repro_2026-10-23/` | Current | Reproduction of the AMER paper with the four data fixes, 3 seeds (`REPORT.md`; plan and status log in `PLAN.md`). Code: tag `paper-repro-v1` |
| `objective_audit_2026-10-13/F_paper_stack_runs_2026-10-27.md` | Current | Session F: INF-Retriever as the query model on the paper stack (code: tag `f-paper-inf-v1`) |
| `objective_audit_2026-10-13/F_paper_gap_plan_2026-10-22.md` | Current | Session F: frozen-doc-encoder multi vs single on QAMPARI, Stages A-F and the seed reruns. Seed tables: `results/phaseC/seed_report.md` |
| `objective_audit_2026-10-13/fusion_combinations_2026-10-16.md` | Current | Fusing several checkpoints' lists |
| `objective_audit_2026-10-13/inference_fusion_2026-10-15.md` | Current | Fusion methods over one model's k query lists (RRF k=60 is the headline) |
| `objective_audit_2026-10-13/PLAN.md` | Current | Objective-audit plan and the shared Status log all sessions append to |
| `weekly_summary_2026-10-12.md` / `.html` / `weekly_summary_data.json` | History | Weekly summary as of 2026-10-12 |
| `wave3_prep_2026-10-09.md` | History | Wave-3 preparation |
| `ablation_clean_v3_2026-10-07.md` | History | Clean v3 ablations; headline k=5 RRF decision |
| `cheap_eval_validity_2026-10-07.md` | History | Whether the cheap reduced-corpus eval ranks runs like the full eval |
| `grid_results_2026-10-02.md` | History | Grid results. Its QAMPARI test numbers are inflated by 44 leaked questions |
| `code_audit_2026-09-28.md`, `code_data_audit_2026-09-27.md`, `leak_examples_G1_G2.md` | History | Late-September code and data audits that found the bugs and leaks fixed since |
| `step5_progress_2026-09-27.md` | History | Joint doc-encoder training: reproduction, root cause of the collapse, the clean recipe |

Untracked working folders under `objective_audit_2026-10-13/` (`copy_shortcut/`, `d5/`, `late_collapse/`, `loss_terms/`, `probe/`) belong to the sessions that made them.
