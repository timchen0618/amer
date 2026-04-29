"""
Retrieve with a finetuned INF retriever checkpoint (same loading as gen_embed_new.py)
and the same FAISS / sharding flow as retrieval_base.py.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import structlog
import torch
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

from src.inference_utils import embed_queries_stella, embed_queries_iterative_retrieval, embed_queries
from src.retrieval_utils import add_passages, load_passages
from retrieval_base import retrieve, load_data

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = structlog.get_logger()

try:
    from training.inf_retriever.src import normalize_text as _normalize_text_mod
except ImportError:
    _normalize_text_mod = None


def normalize_np(x, p=2, dim=1, eps=1e-12):
    norm = np.linalg.norm(x, ord=p, axis=dim, keepdims=True)
    norm = np.maximum(norm, eps)
    return x / norm


def find_checkpoint_dir(model_path: str) -> str | None:
    for candidate in [
        os.path.join(model_path, "checkpoint.pth"),
        os.path.join(model_path, "best_model", "checkpoint.pth"),
    ]:
        if os.path.exists(candidate):
            return os.path.dirname(candidate)
    return None


def load_model_and_tokenizer(args):
    """
    Match gen_embed_new.py: finetuned checkpoints load via load_retriever;
    otherwise HuggingFace / SentenceTransformer paths.
    """
    checkpoint_dir = find_checkpoint_dir(args.model_name_or_path)
    use_finetuned = checkpoint_dir is not None
    tokenizer = None

    if use_finetuned:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "training", "inf_retriever"))
        from src.inference_utils import load_retriever

        print(f"Detected finetuned checkpoint at {checkpoint_dir}, loading via load_retriever")
        model, tokenizer, _ = load_retriever(checkpoint_dir)
    elif (
        ("stella" in args.model_name_or_path)
        or ("inf-retriever" in args.model_name_or_path)
        or ("NV-Embed" in args.model_name_or_path)
    ):
        model = SentenceTransformer(args.model_name_or_path, trust_remote_code=True)
        if "inf-retriever" in args.model_name_or_path:
            model.max_seq_length = 8192
        if "NV-Embed" in args.model_name_or_path:
            model.max_seq_length = 8192
            model.tokenizer.padding_side = "right"
    elif "iterative_retrieval" in args.model_name_or_path:
        model = AutoModel.from_pretrained(args.model_name_or_path)
    else:
        model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    print("finish loading model")
    model.eval()
    model = model.cuda()
    if not args.no_fp16:
        model = model.half()

    return model, tokenizer, use_finetuned


def _maybe_normalize_text(q: str, args) -> str:
    if args.normalize_text and _normalize_text_mod is not None:
        return _normalize_text_mod.normalize(q)
    return q


@torch.no_grad()
def embed_queries_single(args, queries, model, tokenizer):
    def last_token_pool(last_hidden_states: torch.Tensor,
                        attention_mask: torch.Tensor) -> torch.Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def get_detailed_instruct(task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery: {query}'

    def _process_batch(batch):
        batch_dict = tokenizer(batch, max_length=max_length, padding=True, truncation=True, return_tensors='pt')
        for k, v in batch_dict.items():
            batch_dict[k] = v.to(model.device)
        outputs = model(**batch_dict)
        return last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask']).cpu().numpy()

    task = 'Given a web search query, retrieve relevant passages that answer the query'
    model.eval()
    embeddings, batch_question = [], []
    max_length = 1024

    for q in queries:
        if args.lowercase:
            q = q.lower()
        q = _maybe_normalize_text(q, args)
        batch_question.append(get_detailed_instruct(task, q))

        if len(batch_question) == args.per_gpu_batch_size:
            embeddings.append(_process_batch(batch_question))
            batch_question = []

    if batch_question:
        embeddings.append(_process_batch(batch_question))

    embeddings = np.concatenate(embeddings, axis=0)
    embeddings = normalize_np(embeddings, p=2, dim=1)
    print("Questions embeddings shape:", embeddings.shape)
    return embeddings


def aggregate_round_robin(all_results, n_docs):
    """Merge k per-embedding ranked lists into one list per query via round-robin interleaving.

    Documents are added in order: rank-0 from embedding-0, rank-0 from embedding-1, ...,
    rank-0 from embedding-(k-1), rank-1 from embedding-0, ... Duplicates are skipped.
    The score stored for each document is its score from the first embedding that retrieved it.

    Args:
        all_results: list of k result sets, each a list of num_queries (ids, scores) tuples
                     where ids is a list[str] and scores is a numpy array.
        n_docs:      max documents per query in the output.

    Returns:
        list of num_queries ([ids], np.ndarray scores) tuples, same format as retrieve().
    """
    num_embeddings = len(all_results)
    num_queries = len(all_results[0])
    merged = []
    for query_idx in range(num_queries):
        seen_doc_ids = set()
        ids_out = []
        scores_out = []
        deepest_rank = max(len(all_results[emb_idx][query_idx][0]) for emb_idx in range(num_embeddings))
        for rank in range(deepest_rank):
            for emb_idx in range(num_embeddings):
                ids, scores = all_results[emb_idx][query_idx]
                if rank < len(ids):
                    doc_id = ids[rank]
                    if doc_id not in seen_doc_ids:
                        seen_doc_ids.add(doc_id)
                        ids_out.append(doc_id)
                        scores_out.append(float(scores[rank]))
            if len(ids_out) >= n_docs:
                break
        merged.append((ids_out[:n_docs], np.array(scores_out[:n_docs])))
    return merged


def aggregate_rrf(all_results, n_docs, smoothing=60):
    """Merge k per-embedding ranked lists via Reciprocal Rank Fusion (RRF).

    Each document receives an RRF score = sum over all embedding lists of
    1 / (rank + smoothing), where rank is 1-based. Documents that appear in
    multiple lists accumulate higher scores. The merged list is sorted by
    descending RRF score.

    Args:
        all_results: list of k result sets, each a list of num_queries (ids, scores) tuples.
        n_docs:      max documents per query in the output.
        smoothing:   RRF smoothing constant (default 60, as in the original RRF paper).

    Returns:
        list of num_queries ([ids], np.ndarray rrf_scores) tuples, same format as retrieve().
    """
    num_embeddings = len(all_results)
    num_queries = len(all_results[0])
    merged = []
    for query_idx in range(num_queries):
        rrf_scores = {}
        for emb_idx in range(num_embeddings):
            ids, _ = all_results[emb_idx][query_idx]
            for rank, doc_id in enumerate(ids):
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (rank + 1 + smoothing)

        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:n_docs]
        ids_out = [doc_id for doc_id, _ in sorted_docs]
        scores_out = np.array([score for _, score in sorted_docs])
        merged.append((ids_out, scores_out))
    return merged


@torch.no_grad()
def embed_queries_multi(args, queries, model, tokenizer):
    """Generate k query embeddings per question using EmbeddingModelDocEncNoProj.generate().

    Mirrors embed_queries_single but applies left-padding (required by the autoregressive
    model) and calls model.generate() to produce max_new_tokens embeddings per query.

    Args:
        args:     namespace with per_gpu_batch_size, lowercase, normalize_text, max_new_tokens
        queries:  list of raw question strings
        model:    EmbeddingModelDocEncNoProj instance (already on GPU)
        tokenizer: tokenizer matching the model

    Returns:
        np.ndarray of shape (num_queries, max_new_tokens, hidden_dim), L2-normalized.
    """
    def get_detailed_instruct(task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery: {query}'

    task = 'Given a web search query, retrieve relevant passages that answer the query'
    max_length = 1024
    max_new_tokens = getattr(args, 'max_new_tokens', 5)
    device = next(model.parameters()).device

    model.eval()
    all_embeddings = []
    batch_question = []

    def _process_batch(batch):
        encoded = tokenizer(
            batch,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors='pt',
        )
        q_tokens = encoded['input_ids']
        q_mask = encoded['attention_mask'].long()
        bsz, seq_len = q_tokens.shape

        # Convert right-padding → left-padding to match training format
        for i in range(bsz):
            num_pad = (q_mask[i] == 0).sum().item()
            if num_pad > 0:
                q_tokens[i] = torch.cat([
                    q_tokens[i, -num_pad:].clone().fill_(tokenizer.pad_token_id),
                    q_tokens[i, q_mask[i] == 1],
                ])
                q_mask[i] = torch.cat([
                    torch.zeros(num_pad, dtype=torch.long),
                    torch.ones(seq_len - num_pad, dtype=torch.long),
                ])

        q_position_ids = q_mask.cumsum(dim=1) - 1
        q_position_ids = q_position_ids.clamp(min=0)

        # (bsz, max_new_tokens, hidden_dim)
        batch_emb = model.generate(
            q_tokens.to(device),
            q_mask.to(device),
            q_position_ids.to(device),
            max_new_tokens=max_new_tokens,
        )
        return batch_emb.cpu().numpy()

    for q in queries:
        if args.lowercase:
            q = q.lower()
        q = _maybe_normalize_text(q, args)
        batch_question.append(get_detailed_instruct(task, q))

        if len(batch_question) == args.per_gpu_batch_size:
            all_embeddings.append(_process_batch(batch_question))
            batch_question = []

    if batch_question:
        all_embeddings.append(_process_batch(batch_question))

    embeddings = np.concatenate(all_embeddings, axis=0)  # (num_queries, k, hidden_dim)
    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-12)
    print(f"Multi-query embeddings shape: {embeddings.shape}")
    return embeddings


def main(args):
    # Load data
    data_paths = glob.glob(args.data)
    if not data_paths:
        raise AssertionError("No data paths found")
    print("Loading data from data_paths", data_paths)

    # Load model
    model, tokenizer, use_finetuned = load_model_and_tokenizer(args)

    # Load passages
    passages = load_passages(args.passages)
    passage_id_map = {x["id"]: x for x in passages}

    # Inference
    for path in data_paths:
        print("loading data from ", path)
        data = load_data(path)

        output_path = os.path.join(args.output_dir, args.output_file)
        print("start embedding queries")
        queries = [ex.get("question") or ex.get("question_text") or "" for ex in data]

        if use_finetuned:
            assert tokenizer is not None
            if getattr(model, '_is_multi_query', False):
                questions_embedding = embed_queries_multi(args, queries, model, tokenizer)
            else:
                questions_embedding = embed_queries_single(args, queries, model, tokenizer)
        elif (
            ("stella" in args.model_name_or_path)
            or ("inf-retriever" in args.model_name_or_path)
            or ("Qwen" in args.model_name_or_path)
        ):
            questions_embedding = embed_queries_stella(args, queries, model)
        elif 'NV-Embed' in args.model_name_or_path:
            questions_embedding = embed_queries(args, queries, model)
        elif "iterative_retrieval" in args.model_name_or_path:
            questions_embedding = embed_queries_iterative_retrieval(args, queries, model)
        else:
            raise ValueError(
                f"Unsupported model for retrieval_inf.py: {args.model_name_or_path}. "
                "Use a finetuned checkpoint dir, inf-retriever / Qwen / stella ST, or iterative_retrieval HF id."
            )

        print("finished embedding queries")

        if args.save_embeddings:
            np.save(os.path.join(args.output_dir, f"questions_embeddings_{Path(path).stem}.npy"), questions_embedding)
            return

        if use_finetuned and getattr(model, '_is_multi_query', False):
            k_emb = questions_embedding.shape[1]
            all_results = []
            for ki in range(k_emb):
                results_i = retrieve(
                    questions_embedding[:, ki, :],
                    args.num_shards,
                    args.passages_embeddings,
                    passage_id_map,
                    embedding_size=args.projection_size,
                    top_k_per_query=args.n_docs,
                    top_k=args.n_docs,
                    save_or_load_index=args.save_or_load_index,
                    use_gpu=args.use_gpu,
                    n_subquantizers=args.n_subquantizers,
                    n_bits=args.n_bits,
                )
                all_results.append(results_i)
            agg_fn = aggregate_rrf if args.agg_func == "rrf" else aggregate_round_robin
            top_ids_and_scores = agg_fn(all_results, args.n_docs)
        else:
            top_ids_and_scores = retrieve(
                questions_embedding,
                args.num_shards,
                args.passages_embeddings,
                passage_id_map,
                embedding_size=args.projection_size,
                top_k_per_query=args.n_docs,
                top_k=args.n_docs,
                save_or_load_index=args.save_or_load_index,
                use_gpu=args.use_gpu,
                n_subquantizers=args.n_subquantizers,
                n_bits=args.n_bits,
            )

        add_passages(data, passage_id_map, top_ids_and_scores)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as fout:
            for ex in data:
                json.dump(ex, fout, ensure_ascii=False)
                fout.write("\n")
        print(f"Saved results to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        required=True,
        type=str,
        help=".json / .jsonl with question or question_text",
    )
    parser.add_argument("--passages", type=str, default=None, help="Path to passages (.tsv)")
    parser.add_argument("--passages_embeddings", type=str, default=None, help="Glob for encoded passage pickles")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--output_file", type=str, default="output.jsonl")
    parser.add_argument("--n_docs", type=int, default=100)
    parser.add_argument("--per_gpu_batch_size", type=int, default=16)
    parser.add_argument("--save_or_load_index", action="store_true")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--no_fp16", action="store_true")
    parser.add_argument(
        "--question_maxlength",
        type=int,
        default=1024,
        help="Max tokens for query encoding (finetuned iterative path)",
    )
    parser.add_argument(
        "--query_instruct_task",
        type=str,
        default="Given a query, retrieve relevant passages that answer the query",
        help="Instruct line for finetuned checkpoints (matches training/inf_retriever finetuning_data.py)",
    )
    parser.add_argument("--indexing_batch_size", type=int, default=1000000)
    parser.add_argument(
        "--projection_size",
        type=int,
        default=1536,
        help="Embedding dimension (INF-Retriever hidden size is 1536)",
    )
    parser.add_argument("--n_subquantizers", type=int, default=0)
    parser.add_argument("--n_bits", type=int, default=8)
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--normalize_text", action="store_true")
    parser.add_argument("--num_shards", type=int, default=8)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--save_embeddings", action="store_true")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=5,
        help="Number of query embeddings to generate per question (multi-query mode only)",
    )
    parser.add_argument(
        "--agg_func",
        type=str,
        default="round_robin",
        choices=["round_robin", "rrf"],
        help="Aggregation function for multi-query results (multi-query mode only)",
    )
    args = parser.parse_args()
    main(args)
