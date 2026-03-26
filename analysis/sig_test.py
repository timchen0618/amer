from logging import root
import numpy as np
from typing import List
from pathlib import Path

def is_significantly_different(
    scores_A: List,
    scores_B: List,
    alpha: float = 0.05,
    n_trial: int = 10000,
    verbose: bool = False,
) -> bool:
    """Determine if the two lists of model performance are significantly
    different from each other by conducting paired bootstrapping test.

    ! Note: `scores_A` and `scores_B` need to be paired; otherwise, the result is not meaningful.

    Args:
        scores_A (List): First list of score.
        scores_B (List): Second list of score.
        alpha (float, optional): threshold for p-value (below which to be significant). Defaults to 0.05.
        n_trial (int, optional): number of bootstrap sampling to conduct. Defaults to 10000.
        verbose (bool, optional): Whether to print some intermediate results. Defaults to False.

    Returns:
        bool: whether scores_A and scores_B are significantly different from each other.
    """
    scores_A = np.array(scores_A)
    scores_B = np.array(scores_B)
    assert len(scores_A) == len(scores_B)

    # Get the inequality direction (or null hypothesis) we want to validate
    # (by calculating the raw average difference).
    # In this context, let's just call it "the ranking".
    scores_A_mean = scores_A.mean()
    scores_B_mean = scores_B.mean()
    delta = scores_B_mean - scores_A_mean

    count = 0
    n_boostrap = len(scores_A)
    for _ in range(n_trial):
        rand_ids = np.random.choice(len(scores_A), size=n_boostrap, replace=True)
        bootstrapped_scores_A = scores_A[rand_ids]
        bootstrapped_scores_B = scores_B[rand_ids]

        # Count how many times that the bootstrapped average *follows* the ranking
        if delta > 0:
            count += bootstrapped_scores_B.mean() > bootstrapped_scores_A.mean()
        else:
            count += bootstrapped_scores_B.mean() < bootstrapped_scores_A.mean()

    # how many times that the randomness (from bootstrap) causes the ranking to be violated.
    p = 1 - count / n_trial
    # if the amount of violation is below the specified threshold,
    # then it's significant difference.
    is_sig_diff = p <= alpha

    if verbose:
        print(f"Score_A avg: {np.round(scores_A_mean, 1)}")
        print(f"Score_B avg: {np.round(scores_B_mean, 1)}")
        print(f"Delta (B - A): {np.round(delta, 1)}")
        print(f"p: {p} (threshold = {alpha})")
        if is_sig_diff:
            print("Significant")
        else:
            print("*Not* Significant")

    return is_sig_diff


if __name__ == "__main__":
    project_dir = '.'
    rootdir = Path(f'{project_dir}/results/qwen3-4b/qampari_inf/')
    # select_indices_file=f"{project_dir}/data/qampari_5_to_8/large_distance_indices_inf.txt"
    # selected_indices = [int(l.strip('\n').strip('\r')) for l in open(select_indices_file)]
    selected_indices = None
    
    print('======== QAMPARI =========')
    score_type = "mrecall"
    for score_type in ["mrecall"]:
        print('-' * 40)
        print(f"Testing {score_type}")
        base_results = np.load(rootdir / "single" / f"retrieval_out_dev_qampari_5_to_8_max_new_tokens_1_{score_type}_topk100.npy")

        suffix_list = [
            "multi"
        ]
        if selected_indices is not None:
            selected_indices = np.array(selected_indices)
            base_results = base_results[selected_indices]
            
        for suffix in suffix_list:
            results = np.load(rootdir / suffix / f"retrieval_out_dev_qampari_5_to_8_max_new_tokens_5_{score_type}_topk100.npy")
            print(len(results), len(base_results))
            
            if selected_indices is not None:
                results = results[selected_indices]
            is_significantly_different(base_results, results, verbose=True)


    print('======== AMBIGUOUS QE =========')
    rootdir = Path(f'{project_dir}/results/qwen3-4b/ambiguous_qe_inf/')
    # select_indices_file=f"{project_dir}/data/ambiguous/qampari_embeddings_data/large_distance_indices_inf.txt"
    # selected_indices = [int(l.strip('\n').strip('\r')) for l in open(select_indices_file)]
    selected_indices = None
    score_type = "mrecall"
    for score_type in ["mrecall"]:
        print('-' * 40)
        print(f"Testing {score_type}")
        base_results = np.load(rootdir / "single" / f"retrieval_out_dev_ambiguous_qe_max_new_tokens_1_{score_type}_topk100.npy")

        suffix_list = [    
            "multi_SS"
        ]
        if selected_indices is not None:
            selected_indices = np.array(selected_indices)
            base_results = base_results[selected_indices]
        for suffix in suffix_list:
            results = np.load(rootdir / suffix / f"retrieval_out_dev_ambiguous_qe_max_new_tokens_2_{score_type}_topk100.npy")
            print(len(results), len(base_results))
            if selected_indices is not None:
                results = results[selected_indices]
            is_significantly_different(base_results, results, verbose=True)
