import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import transformers
import logging
import torch.distributed as dist

from . import dist_utils, utils


logger = logging.getLogger(__name__)


@torch.no_grad()
def geometry_stats(outputs, positive_embeddings, negative_embeddings):
    """Diagnostics of query/gold geometry for one batch (no gradients, no collectives).

    outputs: (bsz, k, d) query embeddings; positive_embeddings: (bsz, k, d) or (bsz * k, d),
    row j of example i = its j-th gold; negative_embeddings: any (..., d), the local negatives.

    Returns {name: (sum, count)} with sums over examples (per-step names count only examples with
    k > j), so callers can accumulate across batches and divide:
      align_cos / step_{j}_align_cos  cosine of each query embedding to the gold it is matched to by
                                      a Hungarian assignment that maximizes total cosine. This is the
                                      assignment every Hungarian loss here makes (log-softmax only
                                      adds a per-row constant). For k=1, the cosine to the one gold.
      align_margin                    matched-gold cosine minus the largest cosine to any other gold
                                      of the same example (k > 1).
      gold_pair_cos / _min            mean / min cosine between an example's golds (k > 1).
      gold_neg_cos                    mean cosine between an example's golds and all local negatives.
      gold_cohesion                   gold_pair_cos - gold_neg_cos (comparable across doc spaces).
      query_pair_cos / _min           mean / min cosine between an example's k query embeddings (k > 1).
    """
    with torch.autocast(device_type=outputs.device.type, enabled=False):   # fp32 even inside bf16 autocast
        return _geometry_stats(outputs, positive_embeddings, negative_embeddings)


def _geometry_stats(outputs, positive_embeddings, negative_embeddings):
    from scipy.optimize import linear_sum_assignment
    bsz, k, d = outputs.shape
    q = F.normalize(outputs.float(), dim=-1)
    p = F.normalize(positive_embeddings.float().reshape(bsz, k, d), dim=-1)
    n = F.normalize(negative_embeddings.float().reshape(-1, d), dim=-1)
    qp = torch.bmm(q, p.transpose(1, 2))                                   # (bsz, k, k)
    gold_neg = torch.einsum("bkd,nd->bkn", p, n).mean(dim=(1, 2))          # (bsz,)
    if k > 1:
        off = ~torch.eye(k, dtype=torch.bool, device=q.device)
        pp = torch.bmm(p, p.transpose(1, 2))[:, off].view(bsz, -1)        # (bsz, k*(k-1))
        qq = torch.bmm(q, q.transpose(1, 2))[:, off].view(bsz, -1)
        packed = torch.cat([qp.reshape(bsz, -1), gold_neg[:, None], pp.mean(1, keepdim=True),
                            pp.min(1, keepdim=True).values, qq.mean(1, keepdim=True), qq.min(1, keepdim=True).values], 1)
    else:
        packed = torch.cat([qp.reshape(bsz, -1), gold_neg[:, None]], 1)
    packed = packed.float().cpu().numpy()                                   # one device sync
    qp_np = packed[:, : k * k].reshape(bsz, k, k)
    matched = np.empty((bsz, k)); margin = np.empty((bsz, k))
    for i in range(bsz):
        rows, cols = linear_sum_assignment(qp_np[i], maximize=True)
        matched[i, rows] = qp_np[i, rows, cols]
        if k > 1:
            others = qp_np[i].copy(); others[rows, cols] = -np.inf
            margin[i, rows] = qp_np[i, rows, cols] - others[rows].max(1)
    stats = {"align_cos": (matched.mean(1).sum(), bsz), "gold_neg_cos": (packed[:, k * k].sum(), bsz)}
    for j in range(k):
        stats[f"step_{j}_align_cos"] = (matched[:, j].sum(), bsz)
    if k > 1:
        gp, gmin, qm, qmin = (packed[:, k * k + 1 + c] for c in range(4))
        stats.update({
            "align_margin": (margin.mean(1).sum(), bsz),
            "gold_pair_cos": (gp.sum(), bsz), "gold_pair_cos_min": (gmin.sum(), bsz),
            "gold_cohesion": ((gp - packed[:, k * k]).sum(), bsz),
            "query_pair_cos": (qm.sum(), bsz), "query_pair_cos_min": (qmin.sum(), bsz),
        })
    return {name: (float(s), c) for name, (s, c) in stats.items()}


def add_geometry_stats(iter_stats, stats_prefix, outputs, positive_embeddings, negative_embeddings):
    """Log geometry_stats as batch means (iter_stats convention: (value, weight))."""
    for name, (s, c) in geometry_stats(outputs, positive_embeddings, negative_embeddings).items():
        iter_stats[f"{stats_prefix}/{name}"] = (s / c, c)
    return iter_stats


@torch.no_grad()
def copy_stats(outputs, golds, fed_gold, assignment=None):
    """Copy / repeat diagnostics of one training batch (2026-10-13, objective audit D1).

    outputs (bsz, k, d); golds (bsz, k, d), gold j = the one fed in after output j under teacher forcing;
    fed_gold (bsz, k) bool; assignment (bsz, k) gold index matched to each output by the loss (None:
    a cosine Hungarian match, the one every Hungarian loss here makes without fed-gold constraints).

    Over outputs t >= 1 whose input was gold t - 1 (teacher-forced):
      tf_copy_cos        cos(output t, gold t - 1), the gold just fed in;
      tf_best_unfed_cos  largest cos(output t, g) over golds g not fed in before output t;
      tf_copy_match      share matched to gold t - 1 (copying: near 1; chance about 1 / k).
    Over outputs t >= 1 whose input was output t - 1 (free-running):
      free_repeat_cos    cos(output t, output t - 1);
      free_repeat_top1   share whose most similar gold equals output t - 1's most similar gold.
    Returns {name: (sum, count)}; names with no eligible output are left out.
    """
    bsz, k, d = outputs.shape
    if k < 2:
        return {}
    with torch.autocast(device_type=outputs.device.type, enabled=False):
        q = F.normalize(outputs.float(), dim=-1)
        g = F.normalize(golds.float().reshape(bsz, k, d), dim=-1)
        qg = torch.bmm(q, g.transpose(1, 2))                                     # (bsz, k, k)
        if assignment is None:
            from scipy.optimize import linear_sum_assignment
            qg_np = qg.cpu().numpy()
            assignment = torch.empty(bsz, k, dtype=torch.long)
            for i in range(bsz):
                rows, cols = linear_sum_assignment(qg_np[i], maximize=True)
                assignment[i, torch.as_tensor(rows)] = torch.as_tensor(cols, dtype=torch.long)
        assignment = assignment.to(outputs.device)
        t = torch.arange(1, k, device=outputs.device)
        tf = fed_gold[:, :-1]                                                     # (bsz, k-1): input of output t was gold t-1
        free = ~tf
        copy_cos = qg[:, t, t - 1]                                                # (bsz, k-1)
        earlier = torch.tril(torch.ones(k, k, dtype=torch.bool, device=outputs.device), -1)
        fed_before = earlier.unsqueeze(0) & fed_gold.unsqueeze(1)                 # (bsz, k, k): [t, j] gold j fed before t
        best_unfed = qg.masked_fill(fed_before, float("-inf")).max(-1).values[:, 1:]
        copy_match = (assignment[:, 1:] == (t - 1).unsqueeze(0)).float()
        repeat_cos = (q[:, 1:] * q[:, :-1]).sum(-1)
        top1 = qg.argmax(-1)
        repeat_top1 = (top1[:, 1:] == top1[:, :-1]).float()
    out = {}
    n_tf, n_free = int(tf.sum()), int(free.sum())
    if n_tf:
        out["tf_copy_cos"] = (float(copy_cos[tf].sum()), n_tf)
        out["tf_best_unfed_cos"] = (float(best_unfed[tf].sum()), n_tf)
        out["tf_copy_match"] = (float(copy_match[tf].sum()), n_tf)
    if n_free:
        out["free_repeat_cos"] = (float(repeat_cos[free].sum()), n_free)
        out["free_repeat_top1"] = (float(repeat_top1[free].sum()), n_free)
    return out


def add_copy_stats(iter_stats, stats_prefix, outputs, golds, fed_gold, assignment=None):
    """Log copy_stats as means (iter_stats convention: (value, weight))."""
    for name, (s, c) in copy_stats(outputs, golds, fed_gold, assignment).items():
        iter_stats[f"{stats_prefix}/{name}"] = (s / c, c)
    return iter_stats


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.05, normalize_embeddings=True):
        super().__init__()
        self.temperature = temperature
        self.normalize_embeddings = normalize_embeddings
        self.ce_loss = nn.CrossEntropyLoss(reduce=False)

    def forward(self, outputs, positive_embeddings, negative_embeddings, stats_prefix="", iter_stats=None):
        iter_stats = {} if iter_stats is None else iter_stats  # fresh dict per call (audit M1)
        if self.normalize_embeddings:
            outputs = F.normalize(outputs, dim=-1)
            positive_embeddings = F.normalize(positive_embeddings, dim=-1)
            negative_embeddings = F.normalize(negative_embeddings, dim=-1)

        batch_size, k, d = outputs.shape
        labels = torch.arange(batch_size * k).long().to(outputs.device)
        outputs = outputs.view(batch_size * k, d)
        positive_embeddings = positive_embeddings.view(batch_size * k, d)
        negative_embeddings = negative_embeddings.view(batch_size * k, d)
        all_embeddings = torch.cat([positive_embeddings, negative_embeddings], dim=0)

        # handle distributed training: Use var size gather to handle different sizes of embeddings
        var_sizes = dist_utils.get_varsize(all_embeddings)
        start_idx = 0 if dist_utils.get_rank() == 0 else var_sizes[:dist_utils.get_rank()].sum()
        labels = labels + start_idx
        gather_kemb = dist_utils.varsize_gather(all_embeddings)
        
        # original gather
        # gather_kemb = dist_utils.gather(all_embeddings)
        # labels = labels + dist_utils.get_rank() * len(all_embeddings)

        similarity = torch.einsum('bd,cd->bc', outputs / self.temperature, gather_kemb)
        loss = self.ce_loss(similarity, labels)
        
        
        # log stats
        predicted_idx = torch.argmax(similarity, dim=-1)
        
        accuracy = 100 * (predicted_idx == labels).float().mean()
        stdq = torch.std(outputs, dim=0).mean().item()
        stdk = torch.std(all_embeddings, dim=0).mean().item()
        iter_stats[f"{stats_prefix}/accuracy"] = (accuracy, batch_size)
        iter_stats[f"{stats_prefix}/stdq"] = (stdq, batch_size)
        iter_stats[f"{stats_prefix}/stdk"] = (stdk, batch_size)
        
        iter_stats[f"{stats_prefix}/loss"] = (loss.mean().item(), batch_size)
        return loss.mean(), iter_stats


class HungarianContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.05, use_eos=False, normalize_embeddings=True):
        super().__init__()
        from scipy.optimize import linear_sum_assignment
        self.linear_sum_assignment = linear_sum_assignment
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.use_eos = use_eos
        self.normalize_embeddings = normalize_embeddings

    def forward(self, outputs, positive_embeddings, negative_embeddings, stats_prefix="", iter_stats=None):
        if iter_stats is None:
            iter_stats = {}
        loss = self._core_forward(outputs, positive_embeddings, negative_embeddings)
        bsz = outputs.shape[0]
        iter_stats[f"{stats_prefix}/loss"] = (loss.item(), bsz)
        for name, v in self.last_stats.items():
            iter_stats[f"{stats_prefix}/{name}"] = (v, bsz)
        return loss, iter_stats

    def _core_forward(self, outputs, positive_embeddings, negative_embeddings):
        if self.normalize_embeddings:
            outputs = F.normalize(outputs, dim=-1)
            positive_embeddings = F.normalize(positive_embeddings, dim=-1)
            negative_embeddings = F.normalize(negative_embeddings, dim=-1)

        batch_size, k, d = outputs.shape
        outputs = outputs.view(batch_size * k, d)
        positive_embeddings = positive_embeddings.view(batch_size * k, d)
        negative_embeddings = negative_embeddings.view(batch_size * k, d)
        all_embeddings = torch.cat([positive_embeddings, negative_embeddings], dim=0)
        var_sizes = dist_utils.get_varsize(all_embeddings)

        gather_kemb = dist_utils.varsize_gather(all_embeddings)

        similarity = torch.einsum('bd,cd->bc', outputs / self.temperature, gather_kemb)
        similarity = self.log_softmax(similarity)
        losses = []
        assigned_gold = torch.empty(batch_size, k, dtype=torch.long)  # [i, j] = gold matched to output j
        local_start = 0 if dist_utils.get_rank() == 0 else int(var_sizes[:dist_utils.get_rank()].sum())
        for i in range(batch_size):
            start_idx = local_start + k * i
            batch_scores = similarity[k * i:k * (i + 1)]
            cost_matrix = batch_scores[:, start_idx:start_idx + k]

            row_ind, col_ind = self.linear_sum_assignment(cost_matrix.float().detach().cpu().numpy(), maximize=True)
            assigned_gold[i, torch.as_tensor(row_ind)] = torch.as_tensor(col_ind, dtype=torch.long)
            costs = cost_matrix[row_ind, col_ind]
            costs = -(costs)
            losses.append(costs.mean())
        self.last_assignment = assigned_gold.to(outputs.device)
        self.last_stats = _hungarian_accuracy(similarity.detach(), assigned_gold.to(outputs.device), local_start, k)
        return torch.stack(losses).mean()


@torch.no_grad()
def _hungarian_accuracy(similarity, assigned_gold, local_pos_start, k):
    """Training accuracy of a Hungarian loss (2026-10-13; the unmasked losses logged none).

    similarity (b*k, total) over the gathered pool, rows = this rank's outputs; assigned_gold (b, k);
    local_pos_start = column of this rank's first positive. Returns
      accuracy_strict: the row's argmax over the whole pool is its Hungarian-assigned gold;
      accuracy:        the row's argmax is any gold of the same example (relaxed, as hungarian_masked logs).
    """
    bk = similarity.size(0)
    rows = torch.arange(bk, device=similarity.device)
    block_start = local_pos_start + k * (rows // k)
    target = block_start + assigned_gold.reshape(-1)
    pred = similarity.argmax(dim=-1)
    strict = (pred == target).float().mean().item()
    relaxed = ((pred >= block_start) & (pred < block_start + k)).float().mean().item()
    return {"accuracy_strict": 100.0 * strict, "accuracy": 100.0 * relaxed}


class HungarianMaskedContrastiveLoss(nn.Module):
    """Multi-positive contrastive loss with Hungarian assignment and same-example
    positive masking.

    Given ``outputs`` of shape (bsz, k, d) — k auto-regressive query embeddings
    per example — and ``positive_embeddings`` / ``negative_embeddings`` each of
    shape (bsz, k, d) (the j-th row is the j-th gold / random negative for the
    example), this loss:

    1. Uses the scipy linear-sum-assignment to match each of the k predicted
       embeddings to the best-scoring positive *of the same example*. This
       removes the arbitrary "embedding j must match gold j" supervision.
    2. When computing the softmax denominator, masks out the *other* positives
       of the same example so they do not count as in-batch negatives. Only
       the Hungarian-assigned positive remains among the same-example
       candidates; every negative (local + cross-rank) and every positive /
       negative from *other* examples stays in the pool as usual.

    Differences from ``HungarianContrastiveLoss``:
        - HungarianContrastiveLoss takes log_softmax over the full candidate
          pool *before* the Hungarian step, so the other k-1 positives of the
          same example are inside the log-partition and push the loss up even
          when the model has ranked them correctly (false negatives).
        - HungarianContrastiveLoss has no stats / iter_stats plumbing and
          returns only the scalar loss, so it cannot be dropped into
          EmbeddingModelDocEncNoProj without changes.
        - This loss reports both a strict accuracy (argmax == Hungarian
          assignment, over the masked logits) and a relaxed accuracy (argmax
          over the *unmasked* similarity is any same-example positive), which
          is the fair counterpart to the standard_org_q accuracy curve.
    """

    def __init__(self, temperature=0.05, normalize_embeddings=True):
        super().__init__()
        from scipy.optimize import linear_sum_assignment
        self.linear_sum_assignment = linear_sum_assignment
        self.temperature = temperature
        self.normalize_embeddings = normalize_embeddings

    def forward(self, outputs, positive_embeddings, negative_embeddings, stats_prefix="", iter_stats=None):
        iter_stats = {} if iter_stats is None else iter_stats  # fresh dict per call (audit M1)
        if self.normalize_embeddings:
            outputs = F.normalize(outputs, dim=-1)
            positive_embeddings = F.normalize(positive_embeddings, dim=-1)
            negative_embeddings = F.normalize(negative_embeddings, dim=-1)

        batch_size, k, d = outputs.shape
        bk = batch_size * k
        outputs_flat = outputs.reshape(bk, d)
        positive_flat = positive_embeddings.reshape(bk, d)
        negative_flat = negative_embeddings.reshape(bk, d)

        # Order within each rank's gather chunk: [bsz*k positives ; bsz*k negatives]
        all_embeddings = torch.cat([positive_flat, negative_flat], dim=0)
        var_sizes = dist_utils.get_varsize(all_embeddings)
        rank = dist_utils.get_rank()
        start_idx_local = 0 if rank == 0 else int(var_sizes[:rank].sum().item())
        gather_kemb = dist_utils.varsize_gather(all_embeddings)

        # Local positives live at [local_pos_start, local_pos_end) in the gathered tensor.
        # Positives from other ranks belong to different examples, so they are legitimate
        # in-batch negatives and need no masking.
        local_pos_start = start_idx_local
        local_pos_end = start_idx_local + bk

        similarity = torch.einsum('bd,cd->bc', outputs_flat / self.temperature, gather_kemb)  # (bk, total)

        # Hungarian assignment per example, using the (k, k) block of similarity
        # between this example's k predicted embeddings and its k local positives.
        assigned_targets = torch.empty(bk, dtype=torch.long, device=outputs.device)
        assigned_gold = torch.empty(batch_size, k, dtype=torch.long)  # [i, j] = gold matched to output j
        sim_detached = similarity.detach().float().cpu().numpy()
        for i in range(batch_size):
            pos_start = local_pos_start + k * i
            cost = sim_detached[k * i : k * (i + 1), pos_start : pos_start + k]  # (k, k)
            row_ind, col_ind = self.linear_sum_assignment(cost, maximize=True)
            for ri, ci in zip(row_ind, col_ind):
                assigned_targets[k * i + ri] = pos_start + ci
                assigned_gold[i, ri] = int(ci)
        self.last_assignment = assigned_gold.to(outputs.device)  # read by HungarianMaskedDivLoss

        # Mask out same-example positives except the Hungarian-assigned target.
        q_example = torch.arange(bk, device=outputs.device) // k   # (bk,)
        cand_example = torch.arange(bk, device=outputs.device) // k  # example of each local positive slot
        same_example = q_example.unsqueeze(1).eq(cand_example.unsqueeze(0))  # (bk, bk)

        mask = torch.ones_like(similarity, dtype=torch.bool)          # True = keep
        mask[:, local_pos_start:local_pos_end] = ~same_example        # drop all same-example local positives
        mask[torch.arange(bk, device=outputs.device), assigned_targets] = True  # re-enable the assigned one

        masked_similarity = similarity.masked_fill(~mask, float('-inf'))
        loss = F.cross_entropy(masked_similarity, assigned_targets)

        with torch.no_grad():
            strict_correct = (masked_similarity.argmax(dim=-1) == assigned_targets)
            strict_acc = 100.0 * strict_correct.float().mean()

            raw_pred = similarity.argmax(dim=-1)                      # (bk,)
            # argmax falls inside this example's local positive block?
            pos_block_start = local_pos_start + k * q_example
            pos_block_end = pos_block_start + k
            relaxed_correct = (raw_pred >= pos_block_start) & (raw_pred < pos_block_end)
            relaxed_acc = 100.0 * relaxed_correct.float().mean()

            stdq = torch.std(outputs_flat, dim=0).mean().item()
            stdk = torch.std(all_embeddings, dim=0).mean().item()

        iter_stats[f"{stats_prefix}/accuracy"] = (relaxed_acc.item(), batch_size)
        iter_stats[f"{stats_prefix}/accuracy_strict"] = (strict_acc.item(), batch_size)
        iter_stats[f"{stats_prefix}/stdq"] = (stdq, batch_size)
        iter_stats[f"{stats_prefix}/stdk"] = (stdk, batch_size)
        iter_stats[f"{stats_prefix}/loss"] = (loss.item(), batch_size)
        return loss, iter_stats


class HungarianSeqLoss(nn.Module):
    """Hungarian contrastive loss that knows which golds were fed in (2026-10-13, objective audit E1).

    With teacher forcing, gold j is fed in after output j, and the unconstrained Hungarian match lets
    output j + 1 take that same gold: copying the input solves k - 1 of the k slots
    (reports/objective_audit_2026-10-13/PLAN.md). Here:

    1. Output t may not be matched to a gold that was fed in before it (gold j with j < t and
       fed_gold[:, j]). The diagonal t -> t is always allowed, so a full matching always exists.
       Golds produced by the model itself need no rule: the matching is one-to-one, so output t
       never shares a target with an earlier output.
    2. Same-example golds in the softmax denominator, by ``siblings``:
       'negative' every other gold of the example is a negative, as in HungarianContrastiveLoss
                  (with fed_gold=None this loss equals HungarianContrastiveLoss exactly);
       'used'     only the golds already used before step t (fed in, or matched to an earlier
                  output) are negatives; golds still to come are masked out, as in hungarian_masked.
                  Repeating an earlier query is penalised, being close to a later gold is not.

    fed_gold: (bsz, k) bool, [i, j] = gold j of example i was fed in after output j; None = nothing fed.
    """

    uses_feed_info = True
    FORBIDDEN = -1e9

    def __init__(self, temperature=0.05, siblings="negative", normalize_embeddings=True):
        super().__init__()
        assert siblings in ("negative", "used"), siblings
        from scipy.optimize import linear_sum_assignment
        self.linear_sum_assignment = linear_sum_assignment
        self.temperature = temperature
        self.siblings = siblings
        self.normalize_embeddings = normalize_embeddings

    def forward(self, outputs, positive_embeddings, negative_embeddings, stats_prefix="", iter_stats=None,
                fed_gold=None):
        iter_stats = {} if iter_stats is None else iter_stats
        loss = self._core_forward(outputs, positive_embeddings, negative_embeddings, fed_gold=fed_gold)
        bsz = outputs.shape[0]
        iter_stats[f"{stats_prefix}/loss"] = (loss.item(), bsz)
        for name, v in self.last_stats.items():
            iter_stats[f"{stats_prefix}/{name}"] = (v, bsz)
        return loss, iter_stats

    def _core_forward(self, outputs, positive_embeddings, negative_embeddings, fed_gold=None):
        if self.normalize_embeddings:
            outputs = F.normalize(outputs, dim=-1)
            positive_embeddings = F.normalize(positive_embeddings, dim=-1)
            negative_embeddings = F.normalize(negative_embeddings, dim=-1)

        batch_size, k, d = outputs.shape
        bk = batch_size * k
        outputs_flat = outputs.reshape(bk, d)
        all_embeddings = torch.cat([positive_embeddings.reshape(bk, d), negative_embeddings.reshape(bk, d)], dim=0)
        var_sizes = dist_utils.get_varsize(all_embeddings)
        rank = dist_utils.get_rank()
        local_start = 0 if rank == 0 else int(var_sizes[:rank].sum().item())
        gather_kemb = dist_utils.varsize_gather(all_embeddings)
        similarity = torch.einsum('bd,cd->bc', outputs_flat / self.temperature, gather_kemb)  # (bk, total)

        fed = (np.zeros((batch_size, k), dtype=bool) if fed_gold is None
               else fed_gold.detach().cpu().numpy().astype(bool).reshape(batch_size, k))
        sim_np = similarity.detach().float().cpu().numpy()
        assigned = np.empty((batch_size, k), dtype=np.int64)
        keep_block = np.zeros((bk, k), dtype=bool)  # same-example golds kept in row (i, t)'s denominator
        earlier = np.tril(np.ones((k, k), dtype=bool), -1)  # earlier[t, j] = j < t
        for i in range(batch_size):
            ps = local_start + k * i
            cost = sim_np[k * i:k * (i + 1), ps:ps + k].copy()
            forbidden = earlier & fed[i][None, :]  # [t, j]: gold j was fed in before output t
            cost[forbidden] = self.FORBIDDEN
            rows, cols = self.linear_sum_assignment(cost, maximize=True)
            assigned[i, rows] = cols
            if self.siblings == "negative":
                keep_block[k * i:k * (i + 1)] = True
            else:
                used = forbidden.copy()                                  # fed in before t
                for t in range(k):
                    used[t, assigned[i, :t]] = True                      # matched to an earlier output
                    used[t, assigned[i, t]] = True                       # its own target
                keep_block[k * i:k * (i + 1)] = used

        device = outputs.device
        assigned_t = torch.from_numpy(assigned).to(device)
        rows_idx = torch.arange(bk, device=device)
        targets = local_start + k * (rows_idx // k) + assigned_t.reshape(-1)
        mask = torch.ones_like(similarity, dtype=torch.bool)
        block = torch.zeros(bk, bk, dtype=torch.bool, device=device)
        cols = k * (rows_idx // k).unsqueeze(1) + torch.arange(k, device=device).unsqueeze(0)  # (bk, k)
        block.scatter_(1, cols, torch.from_numpy(keep_block).to(device))
        same_example = (rows_idx // k).unsqueeze(1).eq((torch.arange(bk, device=device) // k).unsqueeze(0))
        mask[:, local_start:local_start + bk] = ~same_example | block
        masked_similarity = similarity.masked_fill(~mask, float('-inf'))
        loss = F.cross_entropy(masked_similarity, targets)

        self.last_assignment = assigned_t
        stats = _hungarian_accuracy(similarity.detach(), assigned_t, local_start, k)
        with torch.no_grad():
            stats["accuracy_strict"] = 100.0 * (masked_similarity.argmax(-1) == targets).float().mean().item()
            stats["fed_gold_frac"] = float(fed[:, :-1].mean()) if k > 1 else 0.0
        self.last_stats = stats
        return loss


class HungarianPlusSingleLoss(nn.Module):
    """Multi-query and single-query objectives in the same step (2026-10-04):

        loss = (1 - single_weight) * L_multi + single_weight * L_single

    L_multi is HungarianContrastiveLoss over all k generated embeddings, unchanged.
    L_single trains the first generated embedding (outputs[:, 0], the last-token embedding of the
    query alone, i.e. what a single-query model is) like single-query training does: contrastive
    against ONE gold of the example, drawn uniformly at random each step (single-query training
    draws random.choice over the golds), over the same gathered pool of golds and negatives as
    L_multi. The example's other golds are masked out of the denominator, since single-query
    training never sees them. Both terms gather the pool across ranks in the same order on every
    rank, so the collectives stay matched.
    """

    MULTI_LOSSES = ("hungarian", "hungarian_seq", "hungarian_seq_used")

    def __init__(self, temperature=0.05, single_weight=0.5, normalize_embeddings=True, multi_loss="hungarian"):
        super().__init__()
        assert 0.0 <= single_weight <= 1.0, single_weight
        assert multi_loss in self.MULTI_LOSSES, multi_loss
        # L_multi: HungarianContrastiveLoss (default), or HungarianSeqLoss with the fed-in golds
        # excluded from the matching (--hps_multi_loss, 2026-10-13).
        if multi_loss == "hungarian":
            self.multi = HungarianContrastiveLoss(temperature=temperature, normalize_embeddings=normalize_embeddings)
        else:
            self.multi = HungarianSeqLoss(temperature=temperature, normalize_embeddings=normalize_embeddings,
                                          siblings="negative" if multi_loss == "hungarian_seq" else "used")
        self.uses_feed_info = getattr(self.multi, "uses_feed_info", False)
        self.temperature = temperature
        self.single_weight = single_weight
        self.normalize_embeddings = normalize_embeddings

    def _single(self, query, positive_embeddings, negative_embeddings, k):
        # positives / negatives arrive as (bsz, k, d) or flattened (bsz * k, d), as for L_multi
        if self.normalize_embeddings:
            query = F.normalize(query, dim=-1)
            positive_embeddings = F.normalize(positive_embeddings, dim=-1)
            negative_embeddings = F.normalize(negative_embeddings, dim=-1)
        batch_size, d = query.shape
        all_embeddings = torch.cat([positive_embeddings.reshape(batch_size * k, d),
                                    negative_embeddings.reshape(batch_size * k, d)], dim=0)
        var_sizes = dist_utils.get_varsize(all_embeddings)
        start_idx = 0 if dist_utils.get_rank() == 0 else int(var_sizes[:dist_utils.get_rank()].sum())
        gather_kemb = dist_utils.varsize_gather(all_embeddings)
        similarity = torch.einsum("bd,cd->bc", query / self.temperature, gather_kemb)
        rows = torch.arange(batch_size, device=query.device)
        chosen = torch.randint(0, k, (batch_size,), device=query.device)
        labels = start_idx + rows * k + chosen
        # mask the example's other golds (columns start_idx + i*k + m, m != chosen)
        own = start_idx + rows.unsqueeze(1) * k + torch.arange(k, device=query.device).unsqueeze(0)  # (b, k)
        other = own[torch.arange(k, device=query.device).unsqueeze(0) != chosen.unsqueeze(1)].view(batch_size, k - 1)
        if k > 1:
            similarity = similarity.scatter(1, other, torch.finfo(similarity.dtype).min)
        loss = F.cross_entropy(similarity, labels)
        accuracy = 100 * (similarity.argmax(dim=-1) == labels).float().mean()
        return loss, accuracy

    @property
    def last_assignment(self):
        return self.multi.last_assignment

    def forward(self, outputs, positive_embeddings, negative_embeddings, stats_prefix="", iter_stats=None,
                fed_gold=None):
        iter_stats = {} if iter_stats is None else iter_stats
        if self.uses_feed_info:
            loss_multi = self.multi._core_forward(outputs, positive_embeddings, negative_embeddings, fed_gold=fed_gold)
        else:
            loss_multi = self.multi._core_forward(outputs, positive_embeddings, negative_embeddings)
        loss_single, acc_single = self._single(outputs[:, 0], positive_embeddings, negative_embeddings, outputs.shape[1])
        loss = (1.0 - self.single_weight) * loss_multi + self.single_weight * loss_single
        bsz = outputs.shape[0]
        iter_stats[f"{stats_prefix}/loss_multi"] = (loss_multi.item(), bsz)
        iter_stats[f"{stats_prefix}/loss_single"] = (loss_single.item(), bsz)
        iter_stats[f"{stats_prefix}/accuracy_single"] = (acc_single.item(), bsz)
        for name, v in self.multi.last_stats.items():
            iter_stats[f"{stats_prefix}/multi_{name}"] = (v, bsz)
        iter_stats[f"{stats_prefix}/loss"] = (loss.item(), bsz)
        return loss, iter_stats


class HungarianMaskedDivLoss(HungarianMaskedContrastiveLoss):
    """hungarian_masked plus a penalty on query embeddings that are more alike than their golds (2026-10-07):

        loss = L_masked + div_weight * mean_{i != j} relu(cos(q_i, q_j) - cos(g_a(i), g_a(j)))

    a(.) is L_masked's own Hungarian assignment; the gold-gold cosines are detached. Masking the other
    golds removes the only force that kept the k embeddings apart (in the clean_v3_abl round, masked's
    query embeddings had pairwise cosine 0.94 against 0.57 for their golds). The target is the golds'
    own similarity, not a fixed threshold, because sibling golds are often close; pairs already less
    alike than their golds are not pulled together.
    """

    def __init__(self, temperature=0.05, div_weight=0.5, normalize_embeddings=True):
        super().__init__(temperature=temperature, normalize_embeddings=normalize_embeddings)
        assert div_weight >= 0.0, div_weight
        self.div_weight = div_weight

    def diversity_penalty(self, outputs, positive_embeddings, assignment):
        """outputs (b, k, d); positives (b, k, d) or (b * k, d); assignment (b, k) gold index per output."""
        b, k, d = outputs.shape
        if k < 2:
            return outputs.new_zeros((), dtype=torch.float32)
        q = F.normalize(outputs.float(), dim=-1)
        g = F.normalize(positive_embeddings.detach().float().reshape(b, k, d), dim=-1)
        g = torch.gather(g, 1, assignment.unsqueeze(-1).expand(b, k, d))  # g[:, j] = gold matched to q[:, j]
        qq = q @ q.transpose(1, 2)
        gg = g @ g.transpose(1, 2)
        off = ~torch.eye(k, dtype=torch.bool, device=outputs.device)
        return F.relu(qq - gg)[:, off].mean()

    def forward(self, outputs, positive_embeddings, negative_embeddings, stats_prefix="", iter_stats=None):
        loss_masked, iter_stats = super().forward(outputs, positive_embeddings, negative_embeddings,
                                                  stats_prefix=stats_prefix, iter_stats=iter_stats)
        penalty = self.diversity_penalty(outputs, positive_embeddings, self.last_assignment)
        loss = loss_masked + self.div_weight * penalty
        bsz = outputs.shape[0]
        iter_stats[f"{stats_prefix}/loss_masked"] = (loss_masked.item(), bsz)
        iter_stats[f"{stats_prefix}/div_penalty"] = (penalty.item(), bsz)
        iter_stats[f"{stats_prefix}/loss"] = (loss.item(), bsz)
        return loss, iter_stats


LOSS_REGISTRY = {
    "contrastive": ContrastiveLoss,
    "hungarian_masked": HungarianMaskedContrastiveLoss,
    "hungarian": HungarianContrastiveLoss,
    "hungarian_plus_single": HungarianPlusSingleLoss,
    "hungarian_masked_div": HungarianMaskedDivLoss,
    "hungarian_seq": HungarianSeqLoss,
    "hungarian_seq_used": HungarianSeqLoss,
}


def build_loss(opt):
    """Instantiate the loss module requested by ``opt.loss_fn``.

    ``loss_fn='auto'`` resolves to ``hungarian_masked`` for multi mode and
    ``contrastive`` otherwise, preserving the old default behaviour.
    """
    name = getattr(opt, "loss_fn", "auto")
    if name == "auto":
        name = "hungarian_masked" if getattr(opt, "training_mode", None) == "multi" else "contrastive"
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss_fn={name!r}; valid choices: {sorted(LOSS_REGISTRY)}")
    print(f"[build_loss] training_mode={getattr(opt, 'training_mode', None)!r} -> loss_fn={name!r}", flush=True)
    if name == "hungarian_plus_single":
        if getattr(opt, "training_mode", None) != "multi":
            raise ValueError("loss_fn='hungarian_plus_single' requires training_mode='multi'")
        w = getattr(opt, "single_loss_weight", 0.5)
        multi_loss = getattr(opt, "hps_multi_loss", "hungarian")
        print(f"[build_loss] single_loss_weight={w} hps_multi_loss={multi_loss}", flush=True)
        return HungarianPlusSingleLoss(temperature=opt.temperature, single_weight=w, multi_loss=multi_loss)
    if name in ("hungarian_seq", "hungarian_seq_used"):
        if getattr(opt, "training_mode", None) != "multi":
            raise ValueError(f"loss_fn={name!r} requires training_mode='multi'")
        return HungarianSeqLoss(temperature=opt.temperature, siblings="negative" if name == "hungarian_seq" else "used")
    if name == "hungarian_masked_div":
        if getattr(opt, "training_mode", None) != "multi":
            raise ValueError("loss_fn='hungarian_masked_div' requires training_mode='multi'")
        w = getattr(opt, "div_weight", 0.5)
        print(f"[build_loss] div_weight={w}", flush=True)
        return HungarianMaskedDivLoss(temperature=opt.temperature, div_weight=w)
    return LOSS_REGISTRY[name](temperature=opt.temperature)


class InBatch(nn.Module):
    
    def __init__(self, opt, retriever=None, tokenizer=None):
        super(InBatch, self).__init__()
        
        self.opt = opt
        self.norm_doc = opt.norm_doc
        self.norm_query = opt.norm_query
        self.label_smoothing = opt.label_smoothing
        
        if retriever is None or tokenizer is None:
            retriever, tokenizer = self._load_retriever(
                opt.retriever_model_id, pooling=opt.pooling, random_init=opt.random_init
            )
        self.tokenizer = tokenizer
        self.encoder = retriever
    
    def _load_retriever(self, model_id, pooling, random_init):
        cfg = utils.load_hf(transformers.AutoConfig, model_id, trust_remote_code=True)
        tokenizer = utils.load_hf(transformers.AutoTokenizer, model_id, trust_remote_code=True)

        cfg.name_or_path = model_id

        # Determine model class based on model_id
        if "inf" in model_id.lower() or "infly" in model_id.lower():
            model_class = inf_retriever.INFRetriever
        
        if random_init:
            retriever = model_class(cfg)
        else:
            if "inf" in model_id.lower() or "infly" in model_id.lower():
                pooling = pooling if pooling else "last_token"
                retriever = model_class(cfg, pooling=pooling)
            else:
                retriever = utils.load_hf(model_class, model_id)

        # Set up special tokens for different models
        if "bert-" in model_id:
            if tokenizer.bos_token_id is None:
                tokenizer.bos_token = "[CLS]"
            if tokenizer.eos_token_id is None:
                tokenizer.eos_token = "[SEP]"
        elif "inf" in model_id.lower() or "infly" in model_id.lower():
            pass

        retriever.config.pooling = pooling
        return retriever, tokenizer
    
    def get_encoder(self):
        return self.encoder
    
    def forward(self, q_tokens, q_mask, k_tokens, k_mask, labels=None, stats_prefix="", iter_stats=None, **kwargs):
        iter_stats = {} if iter_stats is None else iter_stats  # fresh dict per call (audit M1)

        bsz = len(q_tokens)
        if labels is not None:  # multiple positives
            len_labels = [len(l) for l in labels]
            labels = torch.tensor([l for ls in labels for l in ls], dtype=torch.long, device=q_tokens.device)
            assert labels.dim() == 1
            len_labels = torch.tensor(len_labels, dtype=torch.long, device=q_tokens.device)            
        else:
            labels = torch.arange(0, bsz, dtype=torch.long, device=q_tokens.device)

        qemb = self.encoder(input_ids=q_tokens, attention_mask=q_mask, normalize=self.norm_query)
        kemb = self.encoder(input_ids=k_tokens, attention_mask=k_mask, normalize=self.norm_doc)
        gather_fn = dist_utils.gather

        gather_kemb = gather_fn(kemb)

        labels = labels + dist_utils.get_rank() * len(kemb)
        
        if labels.size(0) == bsz:
            scores = torch.einsum("id, jd->ij", qemb / self.opt.temperature, gather_kemb)
            loss = torch.nn.functional.cross_entropy(scores, labels, label_smoothing=self.label_smoothing)
        else:
            scores = torch.exp(torch.einsum("id, jd->ij", qemb / self.opt.temperature, gather_kemb))
            num_queries, num_docs = scores.shape
            target_mask = torch.zeros((num_queries, num_docs), device=scores.device)
            negative_mask = torch.ones((num_queries, num_docs), device=scores.device)
            start = 0
            for i, _len in enumerate(len_labels):  # i -> batch index
                pos_indices = labels[start:start+_len]
                target_mask[i, pos_indices] = 1   # Assign positive labels
                negative_mask[i, pos_indices] = 0 # Assign negative labels
                start += _len
            negative_sum = (scores * negative_mask).sum(dim=1, keepdim=True)  # (num_queries, 1)
            positives = (scores * target_mask)  # (num_queries, num_docs)
            contrastive = -torch.log(positives / (positives + negative_sum))  # (num_queries, num_docs)
            loss = contrastive.sum(dim=1) / len_labels
            loss = loss.mean()

        # log stats
        if len(stats_prefix) > 0:
            stats_prefix = stats_prefix + "/"
        iter_stats[f"{stats_prefix}/loss"] = (loss.item(), bsz)

        predicted_idx = torch.argmax(scores, dim=-1)
        
        if labels.size(0) == bsz:
            accuracy = 100 * (predicted_idx == labels).float().mean()
        else:
            start = 0
            accuracy = 0
            for i, _len in enumerate(len_labels):  # i -> batch index
                pos_indices = labels[start:start+_len]
                if predicted_idx[i].item() in pos_indices:
                    accuracy += 1
            accuracy = 100 * (float(accuracy) / len_labels.size(0))
        stdq = torch.std(qemb, dim=0).mean().item()
        stdk = torch.std(kemb, dim=0).mean().item()
        iter_stats[f"{stats_prefix}/accuracy"] = (accuracy, bsz)
        iter_stats[f"{stats_prefix}/stdq"] = (stdq, bsz)
        iter_stats[f"{stats_prefix}/stdk"] = (stdk, bsz)

        return loss, iter_stats





FROZEN_DOC_PREFIX = "doc_encoder."


def load_model_state(model, state_dict):
    """load_state_dict that tolerates a missing frozen doc-encoder copy, and nothing else.

    Frozen-doc-encoder checkpoints (EmbeddingModelFrozenDocEnc*) are saved without their
    doc_encoder.* weights (utils.save_state_dict drops them, 2026-10-09): they are the base
    retriever's, which the class constructor loads anyway. Every other key must match exactly.
    """
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    bad = [k for k in missing if not k.startswith(FROZEN_DOC_PREFIX)] + list(unexpected)
    if bad:
        raise RuntimeError(f"state_dict mismatch (missing/unexpected keys other than the frozen "
                           f"doc encoder): {bad[:10]}{' ...' if len(bad) > 10 else ''}")
    if missing and not hasattr(model, "doc_encoder"):
        raise RuntimeError("doc_encoder.* keys missing but the model has no doc_encoder")
    return missing

def query_attention_kwargs(model):
    """Extra kwargs for the query encoder's forward: is_causal=True under --force_causal, else none.

    The backbone's Qwen2Model.forward (infly modeling_qwen.py) defaults is_causal to False and only
    builds a causal 4-D mask when the caller passes is_causal=True; config.is_causal is not read.
    Passing nothing in the default case keeps the bidirectional runs' calls unchanged.
    """
    return {"is_causal": True} if getattr(model, "force_causal", False) else {}


class EmbeddingModelDocEncNoProj(nn.Module):
    """
    Autoregressive multi-query encoder: one left-padded query sequence produces
    num_query_embeddings query vectors (last-token hidden states), with hidden
    states fed back as additional input tokens (no projection).

    Forward:
        q_tokens, q_mask, q_position_ids: (batch_size, seq_len) — left-padded query
        k_tokens, k_mask: (batch_size, num_query_embeddings * 2, seq_len)
            First num_query_embeddings slots are positive docs, second half negatives.

    Query output shape passed to ContrastiveLoss: (batch_size, num_query_embeddings, dim).
    """

    def __init__(self, opt, retriever=None, tokenizer=None):
        super(EmbeddingModelDocEncNoProj, self).__init__()

        self.opt = opt
        self.norm_doc = opt.norm_doc
        self.norm_query = opt.norm_query
        self.label_smoothing = opt.label_smoothing

        if retriever is None or tokenizer is None:
            retriever, tokenizer = self._load_retriever(
                opt.retriever_model_id
            )
        self.tokenizer = tokenizer
        self.encoder = retriever
        self.embedding = retriever.get_input_embeddings()
        # Loss module is selected via --loss_fn (with 'auto' -> hungarian_masked
        # for multi mode, contrastive otherwise). See build_loss above.
        self.loss_fct = build_loss(opt)
        # --force_causal: causal self-attention in the QUERY encoder (training and generate());
        # documents stay bidirectional. See query_attention_kwargs.
        self.force_causal = getattr(opt, "force_causal", False)
        # Objective-audit options (2026-10-13, reports/objective_audit_2026-10-13/PLAN.md); getattr
        # defaults keep older checkpoints' opts loading unchanged.
        # --detach_fed_output: no gradient through a fed-back output into the step that produced it.
        self.detach_fed_output = getattr(opt, "detach_fed_output", False)
        # --fed_norm token: every fed-in vector (gold or own output, training and generate()) is
        # rescaled to the mean norm of the question's token embeddings.
        self.fed_norm = getattr(opt, "fed_norm", "none")
        assert self.fed_norm in ("none", "token"), self.fed_norm

    def _fed_input(self, vec, q_embeds, q_mask):
        """The vector fed in as the next input token: vec (bsz, 1, d), unchanged unless --fed_norm token."""
        if self.fed_norm == "none":
            return vec
        with torch.no_grad():
            m = q_mask.unsqueeze(-1).to(q_embeds.dtype)
            target = (q_embeds.float().norm(dim=-1, keepdim=True) * m).sum(1, keepdim=True) / m.sum(1, keepdim=True)
        return (F.normalize(vec.float(), dim=-1) * target).to(vec.dtype)

    def _load_retriever(self, model_id):
        tokenizer = utils.load_hf(transformers.AutoTokenizer, model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        retriever = transformers.AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            use_cache=False,
        )
        if hasattr(retriever, 'gradient_checkpointing_enable'):
            retriever.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            print("grad checkpoint enabled for EmbeddingModelDocEncNoProj")
        return retriever, tokenizer

    def get_encoder(self):
        return self.encoder
    
    
    def encode_documents(self, input_document_ids, attention_mask_document):
        """
        Encode document token batches in hidden_size space (no projection).

        - 3D input (batch_size, 2 * nqe, seq_len): returns positives and negatives
          each (batch_size, nqe, hidden_size).
        - 2D input (N, seq_len) with N even (e.g. concat of all golds then all negs):
          returns (N/2, hidden_size) for each half — used by evaluate().
          The first half is all golds, the second half is all negatives.
        """
        # if input_document_ids.dim() == 3:
        #     bsz, num_docs, doc_seq_len = input_document_ids.shape
        #     assert num_docs % 2 == 0, f"num_documents must be even (pos+neg), got {num_docs}"
        #     nqe = num_docs // 2
        #     flat_ids = input_document_ids.reshape(-1, doc_seq_len)
        #     flat_mask = attention_mask_document.reshape(-1, doc_seq_len)
        #     outputs = self.encoder(input_ids=flat_ids, attention_mask=flat_mask)
        #     doc_embeddings = self.last_token_pool(outputs.last_hidden_state, flat_mask)
        #     half = flat_ids.size(0) // 2
        #     positive_embeddings = doc_embeddings[:half, :].view(bsz, nqe, -1)
        #     negative_embeddings = doc_embeddings[half:, :].view(bsz, nqe, -1)
        #     return positive_embeddings, negative_embeddings

        flat_ids = input_document_ids
        flat_mask = attention_mask_document
        batch_size = flat_ids.size(0) // 2
        assert flat_ids.size(0) % 2 == 0, f"num_documents must be even (pos+neg), got {flat_ids.size(0)}"

        outputs = self.encoder(input_ids=flat_ids, attention_mask=flat_mask)
        doc_embeddings = self.last_token_pool(outputs.last_hidden_state, flat_mask)

        positive_embeddings = doc_embeddings[:batch_size, :]
        negative_embeddings = doc_embeddings[batch_size:, :]
        assert positive_embeddings.size(0) == batch_size
        assert negative_embeddings.size(0) == batch_size
        return positive_embeddings, negative_embeddings

    def last_token_pool(self, last_hidden_states, attention_mask):
        """
        Pool the last token as per INF-Retriever documentation
        """
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
        


    @torch.no_grad()
    def generate(self, q_tokens, q_mask, q_position_ids, max_new_tokens=5):
        """Autoregressively generate multiple query embeddings at inference time.

        Each step pools the last-token hidden state as one query embedding, then feeds
        it back as the next input token. No teacher forcing or scheduled sampling.

        Args:
            q_tokens:       (batch_size, seq_len) — left-padded query token ids
            q_mask:         (batch_size, seq_len) — attention mask (1 = real, 0 = pad)
            q_position_ids: (batch_size, seq_len) — position ids for left-padded input
            max_new_tokens: number of query embeddings to generate

        Returns:
            (batch_size, max_new_tokens, hidden_size) float tensor, unnormalized
        """
        bsz = q_tokens.size(0)
        device = q_tokens.device

        assert (q_mask[:, -1] == 1).all(), \
            "Left-padding expected: last token position must always be a real token"

        q_embeds = current_input = self.embedding(q_tokens)
        attn = q_mask
        pos = q_position_ids
        all_outputs = []

        for j in range(max_new_tokens):
            outputs = self.encoder(
                inputs_embeds=current_input,
                attention_mask=attn,
                position_ids=pos,
                **query_attention_kwargs(self),
            )
            next_hidden = self.last_token_pool(outputs.last_hidden_state, attn)
            all_outputs.append(next_hidden.unsqueeze(1))

            if j == max_new_tokens - 1:
                break

            next_tok = self._fed_input(next_hidden.unsqueeze(1), q_embeds, q_mask)
            current_input = torch.cat((current_input, next_tok), dim=1)
            attn = torch.cat(
                (attn, torch.ones(bsz, 1, dtype=attn.dtype, device=device)),
                dim=1,
            )
            pos = torch.cat((pos, pos[:, -1:] + 1), dim=1)

        return torch.cat(all_outputs, dim=1)  # (bsz, max_new_tokens, hidden_size)

    def forward(self, q_tokens, q_mask, q_position_ids, k_tokens, k_mask, labels=None, stats_prefix="", iter_stats=None, **kwargs):
        """
        Args:
            q_tokens: (batch_size, seq_len)
            q_mask:   (batch_size, seq_len)
            q_position_ids: (batch_size, seq_len)
            k_tokens: (batch_size*num_query_embeddings * 2, seq_len)
            k_mask:   (batch_size*num_query_embeddings * 2, seq_len)
            labels:   reserved (optional)
            sampling_rate (kwargs): probability of feeding back the predicted hidden state
                on non-final steps; else teacher (positive doc) embedding. Default 1.0.

        Returns:
            loss, iter_stats
        """
        iter_stats = {} if iter_stats is None else iter_stats  # fresh dict per call (audit M1)
        assert k_tokens.dim() == 2 and k_mask.dim() == 2
        assert k_tokens.size(0) % 2 == 0
        bsz = q_tokens.size(0)
        # assert k_tokens.size(0) == bsz
        
        sampling_rate = kwargs.get("sampling_rate", 1.0)
        iter_stats[f"{stats_prefix}/sampling_rate"] = (sampling_rate, bsz)

        # doc_grad (kwargs, default True): False encodes the documents without gradient, so the shared
        # encoder learns only through the query side (--doc_grad_until, objective audit E3, 2026-10-15).
        doc_grad = kwargs.get("doc_grad", True)
        iter_stats[f"{stats_prefix}/doc_grad"] = (float(doc_grad), bsz)
        if doc_grad:
            positive_embeddings, negative_embeddings = self.encode_documents(k_tokens, k_mask)
        else:
            with torch.no_grad():
                positive_embeddings, negative_embeddings = self.encode_documents(k_tokens, k_mask)
        teacher_embeddings = positive_embeddings.reshape(bsz, -1, positive_embeddings.size(-1))
        negative_embeddings = negative_embeddings.reshape(bsz, -1, negative_embeddings.size(-1))
        output_len = teacher_embeddings.size(1)

        assert (q_mask[:, -1] == 1).all(), (
            "Left-padding expected: last position should always be a real token",
            q_mask[:, -1],
        )

        q_embeds = current_input = self.embedding(q_tokens)
        attn = q_mask
        pos = q_position_ids
        device = q_tokens.device
        all_outputs = []
        fed_gold = torch.zeros(bsz, output_len, dtype=torch.bool, device=device)  # [i, j]: gold j fed after output j

        for j in range(output_len):
            outputs = self.encoder(
                inputs_embeds=current_input,
                attention_mask=attn,
                position_ids=pos,
                **query_attention_kwargs(self),
            )
            next_hidden = self.last_token_pool(outputs.last_hidden_state, attn)
            all_outputs.append(next_hidden.unsqueeze(1))

            # Cosine between output j and gold j, the gold fed in AFTER output j under teacher forcing
            # (golds are in shuffled order, so this is a random sibling gold, not output j's matched
            # target). Was logged as step_{j}_teacher_cos_sim before 2026-10-13.
            with torch.no_grad():
                teacher_j = teacher_embeddings[:, j, :].to(dtype=next_hidden.dtype)
                cos_sim_j = F.cosine_similarity(
                    F.normalize(next_hidden, dim=-1),
                    F.normalize(teacher_j, dim=-1),
                    dim=-1,
                ).mean().item()
                iter_stats[f"{stats_prefix}/step_{j}_next_fed_gold_cos"] = (cos_sim_j, bsz)

            if j == output_len - 1:
                break

            use_predicted = (torch.rand(bsz, 1, 1, device=device) < sampling_rate)
            fed_gold[:, j] = ~use_predicted.view(bsz)
            predicted = next_hidden.unsqueeze(1)
            if self.detach_fed_output:
                predicted = predicted.detach()
            teacher = teacher_embeddings[:, j, :].unsqueeze(1).to(dtype=next_hidden.dtype).detach()
            next_tok = self._fed_input(torch.where(use_predicted, predicted, teacher), q_embeds, q_mask)

            current_input = torch.cat((current_input, next_tok), dim=1)
            attn = torch.cat(
                (attn, torch.ones(bsz, 1, dtype=attn.dtype, device=device)),
                dim=1,
            )
            pos = torch.cat((pos, pos[:, -1:] + 1), dim=1)

        selected_outputs_embeddings = torch.cat(all_outputs, dim=1)  # (bsz, k, d)

        # Pairwise cosine similarity of the k generated embeddings (mode collapse diagnostic)
        with torch.no_grad():
            if output_len > 1:
                outputs_norm = F.normalize(selected_outputs_embeddings.float(), dim=-1)  # (bsz, k, d)
                sim_mat = torch.bmm(outputs_norm, outputs_norm.transpose(1, 2))  # (bsz, k, k)
                eye_mask = torch.eye(output_len, dtype=torch.bool, device=device).unsqueeze(0)
                pairwise_sim = sim_mat.masked_fill(eye_mask, 0.0).sum() / (bsz * output_len * (output_len - 1))
                iter_stats[f"{stats_prefix}/pairwise_cos_sim"] = (pairwise_sim.item(), bsz)

        add_geometry_stats(iter_stats, stats_prefix, selected_outputs_embeddings, positive_embeddings, negative_embeddings)

        feed_kwargs = {"fed_gold": fed_gold} if getattr(self.loss_fct, "uses_feed_info", False) else {}
        loss, iter_stats = self.loss_fct(
            selected_outputs_embeddings,
            positive_embeddings,
            negative_embeddings,
            stats_prefix=stats_prefix,
            iter_stats=iter_stats,
            **feed_kwargs,
        )
        add_copy_stats(iter_stats, stats_prefix, selected_outputs_embeddings, teacher_embeddings, fed_gold,
                       getattr(self.loss_fct, "last_assignment", None))
        return loss, iter_stats



    # def forward(self, q_tokens, q_mask, q_position_ids, k_tokens, k_mask, labels=None, stats_prefix="", iter_stats={}, **kwargs):
    #     """
    #     Args:
    #         q_tokens: (batch_size, num_query_embeddings, seq_len)
    #         q_mask:   (batch_size, num_query_embeddings, seq_len)
    #         k_tokens: (batch_size, num_query_embeddings * 2, seq_len)
    #                   First num_query_embeddings are positives, rest are negatives.
    #         k_mask:   (batch_size, num_query_embeddings * 2, seq_len)
    #         labels:   optional, per-query-embedding labels (same semantics as InBatch)

    #     Returns:
    #         loss, iter_stats
    #     """
    #     bsz = q_tokens.size(0)
    #     nqe = self.num_query_embeddings
        
    #     loss_mask = q_mask.detach().clone()
        
    #     # Step 1: Encode documents (hidden_size space, no projection)
    #     positive_embeddings, negative_embeddings = self.encode_documents(
    #         k_tokens, k_mask
    #     )
    #     labels = positive_embeddings  # teacher forcing targets (hidden_size)
    #     output_len = labels.size(1)
        
    #     # Step 2: Run causal LM on query tokens
    #     input_start_for_output = q_mask.sum(dim=1).max()
    #     assert q_mask[:, input_start_for_output:].sum().item() == 0

    #     initial_embeds = self.embedding(q_tokens)
    #     current_input = initial_embeds[:, :input_start_for_output, :]
    #     outputs = self.encoder(inputs_embeds=initial_embeds, attention_mask=q_mask, position_ids=q_position_ids,)
        
    #     # Step 3: Autoregressive generation with scheduled sampling
    #     # No projection — hidden states are fed back directly
    #     sampling_rate = inputs['sampling_rate']
    #     all_outputs = []
    #     for j in range(output_len):
    #         outputs = self.encoder(
    #             inputs_embeds=current_input,
    #             position_ids=q_position_ids[:, :input_start_for_output + j],
    #             attention_mask=q_mask[:, :input_start_for_output + j],
    #         )

    #         next_emb = outputs.last_hidden_state[:, input_start_for_output + j - 1, :]
    #         all_outputs.append(next_emb.unsqueeze(1))

    #         # Scheduled sampling: predicted hidden state or teacher (doc hidden state)
    #         use_predicted = (torch.rand(current_input.size(0), 1, 1) < sampling_rate).to(current_input.device)
    #         predicted = next_emb.unsqueeze(1)                   # already in hidden_size
    #         teacher = labels[:, j].float().unsqueeze(1)         # also in hidden_size
    #         next_input = torch.where(use_predicted, predicted, teacher)

    #         current_input = torch.cat((current_input, next_input), dim=1)

    #         inputs['position_ids'][:, input_start_for_output + j:] = inputs['position_ids'][:, input_start_for_output + j:] + 1
    #         inputs['attention_mask'][:, input_start_for_output + j] = 1
        
        
        
        
        
        
        

        # # Flatten queries: (batch_size * nqe, seq_len)
        # q_tokens_flat = q_tokens.view(bsz * nqe, -1)
        # q_mask_flat = q_mask.view(bsz * nqe, -1)

        # # Flatten documents: (batch_size * nqe * 2, seq_len)
        # k_tokens_flat = k_tokens.view(bsz * nqe * 2, -1)
        # k_mask_flat = k_mask.view(bsz * nqe * 2, -1)


        










        # # Encode queries and documents with the shared encoder
        # qemb = self.encoder(input_ids=q_tokens_flat, attention_mask=q_mask_flat, normalize=self.norm_query)
        # kemb = self.encoder(input_ids=k_tokens_flat, attention_mask=k_mask_flat, normalize=self.norm_doc)

        # # kemb: (bsz * nqe * 2, dim) — split into positives and negatives
        # kemb_all = kemb.view(bsz, nqe * 2, -1)
        # pos_kemb = kemb_all[:, :nqe, :].reshape(bsz * nqe, -1)   # (bsz * nqe, dim)
        # neg_kemb = kemb_all[:, nqe:, :].reshape(bsz * nqe, -1)   # (bsz * nqe, dim)

        # # Effective batch size: each query embedding is treated as an independent query
        # effective_bsz = bsz * nqe

        # # Gather positive doc embeddings across processes for in-batch negatives
        # gather_fn = dist_utils.gather
        # gather_pos_kemb = gather_fn(pos_kemb)   # (world_size * bsz * nqe, dim)
        # gather_neg_kemb = gather_fn(neg_kemb)   # (world_size * bsz * nqe, dim)

        # # Candidate pool: in-batch positives (used as negatives for other queries) + explicit negatives
        # # gather_kemb: (world_size * bsz * nqe * 2, dim)
        # gather_kemb = torch.cat([gather_pos_kemb, gather_neg_kemb], dim=0)

        # if labels is not None:
        #     len_labels = [len(l) for l in labels]
        #     labels = torch.tensor([l for ls in labels for l in ls], dtype=torch.long, device=q_tokens.device)
        #     assert labels.dim() == 1
        #     len_labels = torch.tensor(len_labels, dtype=torch.long, device=q_tokens.device)
        # else:
        #     # Each query embedding's positive is at the corresponding index in gather_pos_kemb
        #     labels = torch.arange(0, effective_bsz, dtype=torch.long, device=q_tokens.device)

        # labels = labels + dist_utils.get_rank() * effective_bsz

        # if labels.size(0) == effective_bsz:
        #     scores = torch.einsum("id, jd->ij", qemb / self.opt.temperature, gather_kemb)
        #     loss = torch.nn.functional.cross_entropy(scores, labels, label_smoothing=self.label_smoothing)
        # else:
        #     scores = torch.exp(torch.einsum("id, jd->ij", qemb / self.opt.temperature, gather_kemb))
        #     num_queries, num_docs = scores.shape
        #     target_mask = torch.zeros((num_queries, num_docs), device=scores.device)
        #     negative_mask = torch.ones((num_queries, num_docs), device=scores.device)
        #     start = 0
        #     for i, _len in enumerate(len_labels):
        #         pos_indices = labels[start:start+_len]
        #         target_mask[i, pos_indices] = 1
        #         negative_mask[i, pos_indices] = 0
        #         start += _len
        #     negative_sum = (scores * negative_mask).sum(dim=1, keepdim=True)
        #     positives = (scores * target_mask)
        #     contrastive = -torch.log(positives / (positives + negative_sum))
        #     loss = contrastive.sum(dim=1) / len_labels
        #     loss = loss.mean()

        # # log stats
        # if len(stats_prefix) > 0:
        #     stats_prefix = stats_prefix + "/"
        # iter_stats[f"{stats_prefix}/loss"] = (loss.item(), bsz)

        # predicted_idx = torch.argmax(scores, dim=-1)

        # if labels.size(0) == effective_bsz:
        #     accuracy = 100 * (predicted_idx == labels).float().mean()
        # else:
        #     start = 0
        #     accuracy = 0
        #     for i, _len in enumerate(len_labels):
        #         pos_indices = labels[start:start+_len]
        #         if predicted_idx[i].item() in pos_indices:
        #             accuracy += 1
        #     accuracy = 100 * (float(accuracy) / len_labels.size(0))
        # stdq = torch.std(qemb, dim=0).mean().item()
        # stdk = torch.std(kemb, dim=0).mean().item()
        # iter_stats[f"{stats_prefix}/accuracy"] = (accuracy, bsz)
        # iter_stats[f"{stats_prefix}/stdq"] = (stdq, bsz)
        # iter_stats[f"{stats_prefix}/stdk"] = (stdk, bsz)

        # return loss, iter_stats
    

class EmbeddingModelDocEncNoProjSingleQuery(nn.Module):
    """
    Multi-query variant of InBatch. Uses the same shared encoder for queries and
    documents, but each query produces num_query_embeddings embeddings.

    Query output shape:  (batch_size, num_query_embeddings, embedding_dim)
    Document input:      num_query_embeddings positive + num_query_embeddings negative
                         docs per sample, encoded with the same encoder.

    Forward signature mirrors InBatch:
        q_tokens:  (batch_size, num_query_embeddings, seq_len)
        k_tokens:  (batch_size, num_query_embeddings * 2, seq_len)
                   first num_query_embeddings are positives, rest are negatives
    """

    def __init__(self, opt, retriever=None, tokenizer=None):
        super(EmbeddingModelDocEncNoProjSingleQuery, self).__init__()

        self.opt = opt
        self.norm_doc = opt.norm_doc
        self.norm_query = opt.norm_query
        self.label_smoothing = opt.label_smoothing

        if retriever is None or tokenizer is None:
            retriever, tokenizer = self._load_retriever(
                opt.retriever_model_id
            )
        self.tokenizer = tokenizer
        self.encoder = retriever
        self.embedding = retriever.get_input_embeddings()
        self.loss_fct = ContrastiveLoss(
            temperature=opt.temperature,
        )
        self.force_causal = getattr(opt, "force_causal", False)  # query encoder only; see query_attention_kwargs

    def _load_retriever(self, model_id):
        tokenizer = utils.load_hf(transformers.AutoTokenizer, model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        retriever = transformers.AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            use_cache=False,
        )
        if hasattr(retriever, 'gradient_checkpointing_enable'):
            retriever.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            print("grad checkpoint enabled for EmbeddingModelDocEncNoProj")
        return retriever, tokenizer

    def get_encoder(self):
        return self.encoder
    
    
    def encode_documents(self, input_document_ids, attention_mask_document):
        """
        Encode raw document tokens using the shared base_causallm (no projection).

        Returns embeddings in hidden_size space.
        """
        flat_ids = input_document_ids
        flat_mask = attention_mask_document
        batch_size = flat_ids.size(0) // 2
        assert flat_ids.size(0) % 2 == 0, f"num_documents must be even (pos+neg), got {flat_ids.size(0)}"

        outputs = self.encoder(input_ids=flat_ids, attention_mask=flat_mask)
        # Last-token pooling → stays in hidden_size space, no projection
        doc_embeddings = self.last_token_pool(outputs.last_hidden_state, flat_mask)

        positive_embeddings = doc_embeddings[:batch_size, :] # first half of the embeddings are positive
        negative_embeddings = doc_embeddings[batch_size:, :] # second half of the embeddings are negative
        assert positive_embeddings.size(0) == batch_size
        assert negative_embeddings.size(0) == batch_size
        return positive_embeddings, negative_embeddings

    def last_token_pool(self, last_hidden_states, attention_mask):
        """
        Pool the last token as per INF-Retriever documentation
        """
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
        


    def forward(self, q_tokens, q_mask, q_position_ids, k_tokens, k_mask, labels=None, stats_prefix="", iter_stats=None, **kwargs):
        """
        Args:
            q_tokens: (batch_size, seq_len)
            q_mask:   (batch_size, seq_len)
            q_position_ids: (batch_size, seq_len)
            k_tokens: (batch_size, 1 * 2, seq_len)
                      First 1 are positives, rest are negatives.
            k_mask:   (batch_size, 1 * 2, seq_len)
            labels:   optional, per-query-embedding labels (same semantics as InBatch)

        Returns:
            loss, iter_stats
        """
        iter_stats = {} if iter_stats is None else iter_stats  # fresh dict per call (audit M1)
        bsz = q_tokens.size(0)
        nqe = 1
        
        
        loss_mask = q_mask.detach().clone()
        
        # Step 1: Encode documents (hidden_size space, no projection)
        positive_embeddings, negative_embeddings = self.encode_documents(k_tokens, k_mask)
        labels = positive_embeddings  # teacher forcing targets (hidden_size)
        output_len = labels.size(1)
        
        # Step 2: Run causal LM on query tokens (left-padded)
        # With left-padding, padding is at the start; real tokens are contiguous at the end.
        # Verify: for each sample, leading positions are padding (0) and trailing are real (1).
        
        # input_start_for_output = q_mask.sum(dim=1).max()
        # assert q_mask[:, input_start_for_output:].sum().item() == 0
        seq_len = q_mask.size(-1)
        num_real = q_mask.sum(dim=-1)  # (bsz, 1)
        pad_len = seq_len - num_real   # (bsz, 1) — number of leading pad tokens per sample
        assert (q_mask[:, -1] == 1).all(), ("Left-padding expected: last position should always be a real token", q_mask[:, -1])

        initial_embeds = self.embedding(q_tokens)
        # current_input = initial_embeds[:, :input_start_for_output, :] 
        outputs = self.encoder(inputs_embeds=initial_embeds, attention_mask=q_mask, position_ids=q_position_ids,
                               **query_attention_kwargs(self))
        # last_token_pool handles left-padding: simply take the last position
        selected_outputs_embeddings = self.last_token_pool(outputs.last_hidden_state, q_mask)
        selected_outputs_embeddings = selected_outputs_embeddings.unsqueeze(1)
        # Step 6: Contrastive loss (all in hidden_size space)
        add_geometry_stats(iter_stats, stats_prefix, selected_outputs_embeddings, positive_embeddings, negative_embeddings)
        loss, iter_stats = self.loss_fct(selected_outputs_embeddings, positive_embeddings, negative_embeddings, stats_prefix=stats_prefix, iter_stats=iter_stats)
        return loss, iter_stats


def _maybe_apply_lora(encoder, opt):
    """Wrap `encoder` (the trainable query encoder) with a LoRA adapter when
    opt.use_lora is set, matching the paper's real-data recipe (Appendix A.6):
    rank 64, alpha 16, dropout 0.1, applied to all major linear layers. Only
    called on the trainable self.encoder -- the frozen self.doc_encoder never
    gets LoRA, since it's not trained at all (see EmbeddingModelFrozenDocEnc).

    Isolates whether full fine-tuning (no LoRA) of the already-strong
    pretrained infly backbone is itself degrading retrieval quality,
    independent of the multi-query objective (see experiment_plan.md).
    """
    if not getattr(opt, "use_lora", False):
        return encoder
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=getattr(opt, "lora_r", 64),
        lora_alpha=getattr(opt, "lora_alpha", 16),
        lora_dropout=getattr(opt, "lora_dropout", 0.1),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"],
        bias="none",
    )
    return get_peft_model(encoder, lora_config)


class EmbeddingModelFrozenDocEnc(EmbeddingModelDocEncNoProj):
    """
    Same autoregressive multi-query mechanism as EmbeddingModelDocEncNoProj,
    but documents are encoded by a second, separate, frozen copy of the
    backbone (matching the paper's frozen-document-encoder design) instead
    of the trainable query-encoder weights. The query encoder's self-attention
    can optionally be forced causal via opt.force_causal.

    It overrides only __init__ and encode_documents; forward() and generate() are inherited from
    EmbeddingModelDocEncNoProj (2026-10-09; they used to be copies), including --force_causal,
    which the base class now applies to every query-encoder call (query_attention_kwargs).

    NOTE on is_causal: the base retriever's modeling code
    (Qwen2Model.forward) defaults `is_causal` to False and only builds a
    causal attention mask when the caller passes `is_causal=True` explicitly
    on the call; `config.is_causal` in config.json is not read anywhere in
    that forward() and has no effect. So forcing causal masking must be done
    via this per-call kwarg, not by setting an attribute on the loaded model.
    """

    def __init__(self, opt, retriever=None, tokenizer=None):
        super().__init__(opt, retriever, tokenizer)
        self.encoder = _maybe_apply_lora(self.encoder, opt)
        self.embedding = self.encoder.get_input_embeddings()
        doc_encoder, _ = self._load_retriever(opt.retriever_model_id)
        doc_encoder.eval()
        for p in doc_encoder.parameters():
            p.requires_grad_(False)
        self.doc_encoder = doc_encoder

    def encode_documents(self, input_document_ids, attention_mask_document):
        """Same as EmbeddingModelDocEncNoProj.encode_documents, but routed
        through the frozen self.doc_encoder under no_grad instead of the
        trainable self.encoder."""
        flat_ids = input_document_ids
        flat_mask = attention_mask_document
        batch_size = flat_ids.size(0) // 2
        assert flat_ids.size(0) % 2 == 0, f"num_documents must be even (pos+neg), got {flat_ids.size(0)}"

        with torch.no_grad():
            outputs = self.doc_encoder(input_ids=flat_ids, attention_mask=flat_mask)
            doc_embeddings = self.last_token_pool(outputs.last_hidden_state, flat_mask)

        positive_embeddings = doc_embeddings[:batch_size, :]
        negative_embeddings = doc_embeddings[batch_size:, :]
        assert positive_embeddings.size(0) == batch_size
        assert negative_embeddings.size(0) == batch_size
        return positive_embeddings, negative_embeddings

class EmbeddingModelFrozenDocEncSingleQuery(EmbeddingModelDocEncNoProjSingleQuery):
    """
    Single-query baseline reference for Phase 1: same as
    EmbeddingModelDocEncNoProjSingleQuery, but documents are encoded by a
    second, separate, frozen copy of the backbone instead of the trainable
    query-encoder weights.

    --force_causal applies to its single query-encoder pass as well (inherited forward(), 2026-10-09),
    so a causal multi-query run has a causal single-query partner.
    """

    def __init__(self, opt, retriever=None, tokenizer=None):
        super().__init__(opt, retriever, tokenizer)
        self.encoder = _maybe_apply_lora(self.encoder, opt)
        self.embedding = self.encoder.get_input_embeddings()
        doc_encoder, _ = self._load_retriever(opt.retriever_model_id)
        doc_encoder.eval()
        for p in doc_encoder.parameters():
            p.requires_grad_(False)
        self.doc_encoder = doc_encoder

    def encode_documents(self, input_document_ids, attention_mask_document):
        """Same as EmbeddingModelDocEncNoProjSingleQuery.encode_documents, but
        routed through the frozen self.doc_encoder under no_grad."""
        flat_ids = input_document_ids
        flat_mask = attention_mask_document
        batch_size = flat_ids.size(0) // 2
        assert flat_ids.size(0) % 2 == 0, f"num_documents must be even (pos+neg), got {flat_ids.size(0)}"

        with torch.no_grad():
            outputs = self.doc_encoder(input_ids=flat_ids, attention_mask=flat_mask)
            doc_embeddings = self.last_token_pool(outputs.last_hidden_state, flat_mask)

        positive_embeddings = doc_embeddings[:batch_size, :]
        negative_embeddings = doc_embeddings[batch_size:, :]
        assert positive_embeddings.size(0) == batch_size
        assert negative_embeddings.size(0) == batch_size
        return positive_embeddings, negative_embeddings