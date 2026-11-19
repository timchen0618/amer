# archive/

One-off root scripts retired on 2026-11-19, kept for the record. Before the move, a grep found nothing in the repo that runs or imports any of them.
They are not maintained. Several assume the repo root as the working directory or import the root `src/`, so they may not run from here.

- `add_prefix.sh`, `submit_eval_jobs.sh`: identical copies are in `scripts/`.
- `run_eval.sh` and `write_retrieve.py` call each other.
- `launch.py` and `qampari+ambiguous_qe_config.sh` are old sweep launchers for `train_distributed.py`.
- The rest are single-purpose scripts: `aggregate*.py`, `mmlf.py`, `bbb.py`, `eval_limit.py`, `eval_nq_msmarco.py`, `get_single_answer_data.py`, `hyperparameter_search.sh`, `inference_llama_vllm.py`, `length_labels.py`, `massive_store.py`, `mess_with_docs.py`, `mteb_embed.py`, `multiple_or_single.py`, `oracle.py`, `remove.py`, `remove_cache_files.py`, `similarity.py`, `track.py`.
