"""Ranker+MLP decoding with the exact greedy Feature set coupling.

The Target tree runtime supplies logits for all rows at once and compacts only
one accepted path. This decoder owns proposal, rejection, recovery, and EOS
logic; those Python decisions remain outside the captured Target forward.
"""
from __future__ import annotations

from greedy_coupling import candidates as greedy_candidates, select as greedy_select
from univer_select import select as univer_select

import time
from contextlib import nullcontext



def _decode_impl(
    target,
    adapter,
    ranker,
    ranks,
    prompt,
    budget,
    seed,
    *,
    tree_runtime,
    width=2,
    cache_factory=None,
):
    """Decode one sequence and return ``(stats, generated_suffix)``.

    ``tree_runtime.verify`` evaluates ``[pendingF, P1, F_a, P2_a, ...]`` and
    leaves the logical cache position unchanged. ``commit`` keeps the chosen
    root-to-node path. A single accepted Feature branch uses its own Target P2
    row and pre-finalnorm hidden state.
    """
    import torch

    from draft_a.pf_feature_unified import EOS, F_VOCAB, P_VOCAB, _shared
    from bench_gaussiangpt_pf_block2_cow import (
        CaptureLastBlock,
        decide,
        fork_position_state,
        residual_point,
        truncate,
    )
    from gpt_position_spec_decode_smoke import choose_position

    shared = _shared()
    require = shared._require
    ratio = shared._ratio
    width = int(width)
    budget = int(budget)
    if budget < 2 or budget % 2:
        raise ValueError("budget must be a positive even number of at least two")
    if width < 1 or width > int(getattr(tree_runtime, "max_width", width)):
        raise ValueError("width exceeds the tree runtime capacity")
    if width > F_VOCAB:
        raise ValueError("width exceeds the Feature vocabulary")

    prompt = [int(token) for token in prompt]
    require(bool(prompt), "Prompt must contain BOS")
    shared.validate_sequence(prompt, [], ranks)
    require((len(prompt) - 1) % 2 == 0, "Prompt must end at a complete pair")
    require(EOS not in prompt[1:], "Prompt must not contain terminal EOS")

    device = next(target.parameters()).device
    rng = torch.Generator(device=device)
    rng.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()

    stats = {
        "target_calls": 0,
        "target_input_tokens": 0,
        "draft_calls": 0,
        "draft_input_tokens": 0,
        "ranker_calls": 0,
        "rounds": 0,
        "proposed": 0,
        "verified": 0,
        "accepted": 0,
        "accepted_lengths": [],
        "round_generated_lengths": [],
        "position_fallbacks": 0,
        "k1_steps": 0,
        "tree_verify_calls": 0,
        "candidate_tree_nodes": 0,
        "target_tree_nodes": 0,
        "feature_attempts": 0,
        "feature_branch_hits": 0,
        "feature_all_rejected": 0,
        "feature_branch_index_counts": {},
        "recovery_calls": 0,
        "feature2_eligible_rounds": 0,
        "feature2_attempts": 0,
        "feature2_hits": 0,
        "feature2_random_accepts": 0,
        "feature2_residual_hits": 0,
        "feature2_distinct_candidates": 0,
        "feature2_branch_nodes": 0,
        "greedy_candidate_set": True,
        "univer_conditional_tree": True,
        "masked_feature_levels": 2,
        "random_proposal_accepts": 0,
        "residual_set_hits": 0,
        "candidate_width_counts": {},
    }
    for kind in shared.KINDS:
        for field in ("proposed", "verified", "accepted"):
            stats[kind + "_" + field] = 0

    def count(kind, field, amount=1):
        amount = int(amount)
        stats[field] += amount
        stats[kind + "_" + field] += amount

    def policy(logits):
        return shared.policy(logits)

    def draw(logits):
        return int(shared.draw(policy(logits), rng))

    def finish_round(begin, accepted):
        stats["accepted_lengths"].append(int(accepted))
        stats["round_generated_lengths"].append(len(prefix) - begin)

    def validate_position(previous, token):
        require(0 <= int(token) < P_VOCAB, "Target emitted an invalid Position id")
        if previous is not None:
            require(int(ranks[token]) > int(ranks[previous]),
                    "Target emitted a non-increasing Position")

    cache = None
    capture = None
    prefix = list(prompt)
    adapter._profile = None
    capacity = len(prompt) + budget + 2 + 2 * width
    make_cache = cache_factory or shared.cache_for
    try:
        cache = make_cache(target, capacity)
        capture = CaptureLastBlock(target)
        adapter.prepare(prompt, capacity)

        def target_forward(tokens, *, recovery=False):
            tokens = [int(token) for token in tokens]
            if not tokens:
                raise ValueError("Target forward requires at least one token")
            start = int(cache.get_pos())
            kinds = shared.typed_kinds(start, len(tokens))
            logits, states = capture.run(target, cache, tokens, kinds, device)
            if int(cache.get_pos()) != start + len(tokens):
                raise RuntimeError("ordinary Target call advanced the wrong number of tokens")
            stats["target_calls"] += 1
            stats["target_input_tokens"] += len(tokens)
            if recovery:
                stats["recovery_calls"] += 1
            return logits, states

        def verify_tree(pending_f, p1, features, p2s, second_features=None, third_position=None):
            before = int(cache.get_pos())
            extra = {} if second_features is None else dict(second_features=second_features,third_position=third_position)
            logits, states = tree_runtime.verify(cache,pending_f,p1,features,p2s,**extra)
            if int(cache.get_pos()) != before:
                raise RuntimeError("tree verify advanced the logical cache position")
            expected_nodes = 2+2*len(features) if second_features is None else 2+10*len(features)
            if logits.ndim == 3:
                logits = logits[0]
            if states.ndim == 3:
                states = states[0]
            if logits.ndim != 2 or states.ndim != 2:
                raise RuntimeError("tree runtime must return rank-2 logits and hidden states")
            if logits.size(0) != expected_nodes or states.size(0) != expected_nodes:
                raise RuntimeError("tree runtime returned a different node count")
            stats["target_calls"] += 1
            stats["target_input_tokens"] += expected_nodes
            stats["tree_verify_calls"] += 1
            stats["candidate_tree_nodes"] += len(features)
            if second_features is not None:
                stats["feature2_eligible_rounds"] += 1
                stats["feature2_distinct_candidates"] += len(set(x for row in second_features for x in row))
                stats["feature2_branch_nodes"] += 4*len(features)
            stats["target_tree_nodes"] += expected_nodes
            return logits, states

        def commit_path(before, path):
            after = tree_runtime.commit(cache, path)
            expected = int(before) + len(path)
            if int(after) != expected or int(cache.get_pos()) != expected:
                raise RuntimeError("tree commit left the Target cache at an unexpected position")
            return after

        # A complete prompt leaves its final Feature pending. The Target cache
        # and h_prev stop at the preceding Position, matching the MLP adapter.
        online = None
        h_prev = None
        if len(prompt) > 1:
            _, states = target_forward(prompt[:-1])
            h_prev = states[-1].detach().clone()
            online = shared.position_state(prefix, ranks)

        # BOS-only prompts use the same Target P/F bootstrap as the reference.
        while len(prefix) - len(prompt) < budget:
            if online is None:
                logits, _ = target_forward(prefix[-1:])
                p0 = draw(logits[-1])
                prefix.append(p0)
                if p0 == EOS:
                    break
                validate_position(None, p0)
                logits, states = target_forward([p0])
                f0 = draw(logits[-1])
                require(P_VOCAB <= f0 < EOS, "Target emitted an invalid Feature id")
                prefix.append(f0)
                h_prev = states[-1].detach().clone()
                online = shared.position_state(prefix, ranks)
                continue

            if int(cache.get_pos()) != len(prefix) - 1:
                raise RuntimeError("Target cache does not end immediately before pending Feature")
            stats["rounds"] += 1
            begin = len(prefix)
            before = int(cache.get_pos())
            previous, pending_f = prefix[-2], prefix[-1]
            remaining = budget - (len(prefix) - len(prompt))

            stats["ranker_calls"] += 1
            first = choose_position(ranker, online, previous)
            second = shadow = None
            can_tree = first is not None and remaining >= 4
            if can_tree:
                shadow = fork_position_state(online)
                shadow.finalize(first[0])
                stats["ranker_calls"] += 1
                second = choose_position(ranker, shadow, first[0])
                can_tree = second is not None

            if not can_tree:
                stats["k1_steps"] += 1
                if first is None:
                    stats["position_fallbacks"] += 1
                    logits, _ = target_forward([pending_f])
                    actual_p1 = draw(logits[-1])
                    prefix.append(actual_p1)
                    if actual_p1 == EOS:
                        online.drop_eos()
                        finish_round(begin, 0)
                        break
                    validate_position(previous, actual_p1)
                    logits, states = target_forward([actual_p1])
                    actual_f1 = draw(logits[-1])
                    require(P_VOCAB <= actual_f1 < EOS,
                            "Target emitted an invalid Feature id")
                    prefix.append(actual_f1)
                    h_prev = states[-1].detach().clone()
                    online.finalize(actual_p1)
                    finish_round(begin, 0)
                    continue

                p1 = int(first[0])
                count("position", "proposed")
                logits, states = target_forward([pending_f, p1])
                p1_distribution = policy(logits[0])
                count("position", "verified")
                if bool(decide(float(p1_distribution[p1].item()), rng, device)):
                    count("position", "accepted")
                    actual_p1 = p1
                    h_prev = states[1].detach().clone()
                    actual_f1 = draw(logits[1])
                    accepted = 1
                else:
                    actual_p1 = int(residual_point(p1_distribution, p1, rng))
                    truncate(cache, before + 1, before + 2)
                    if int(cache.get_pos()) != before + 1:
                        raise RuntimeError("K1 Position recovery left a bad cache length")
                    if actual_p1 == EOS:
                        prefix.append(actual_p1)
                        online.drop_eos()
                        finish_round(begin, 0)
                        break
                    validate_position(previous, actual_p1)
                    logits, states = target_forward([actual_p1], recovery=True)
                    actual_f1 = draw(logits[-1])
                    h_prev = states[-1].detach().clone()
                    accepted = 0
                require(P_VOCAB <= actual_f1 < EOS,
                        "Target emitted an invalid Feature id")
                prefix.extend([actual_p1, actual_f1])
                online.finalize(actual_p1)
                finish_round(begin, accepted)
                continue

            p1 = int(first[0])
            p2 = int(second[0])
            third = None
            if remaining >= 6:
                shadow2 = fork_position_state(shadow)
                shadow2.finalize(p2)
                stats["ranker_calls"] += 1
                third = choose_position(ranker,shadow2,p2)
            can_long = third is not None
            p3 = int(third[0]) if can_long else None
            count("position", "proposed", 3 if can_long else 2)
            
            adapter.next_position = p2
            q = adapter.propose(prefix, p1, h_prev, rng)
            require(torch.is_tensor(q) and q.dtype == torch.float32 and q.ndim == 1
                    and q.numel() == F_VOCAB,
                    "Feature proposal must be a full FP32 distribution of size 4096")
            q_mass = float(q.sum().item())
            require(abs(q_mass - 1.0) <= 1e-4,
                    "Feature proposal must be normalized after temperature/top-p")
            candidate_ids, candidate_q = greedy_candidates(q, width, rng)
            actual_width = len(candidate_ids)
            count("feature", "proposed", actual_width)
            key = str(actual_width)
            stats["candidate_width_counts"][key] = stats["candidate_width_counts"].get(key, 0) + 1
            features = [int(token) + P_VOCAB for token in candidate_ids]
            p2s = [p2] * actual_width

            second_ids = second_q = second_features = None
            if can_long:
                choices=[greedy_candidates(adapter.second_distribution,4,rng) for _ in candidate_ids]
                second_ids=[item[0] for item in choices]
                second_q=[item[1] for item in choices]
                if any(len(ids)!=4 for ids in second_ids):
                    raise ValueError('Four full-support second candidates required per parent')
                second_features=[[int(token)+P_VOCAB for token in ids] for ids in second_ids]
                count("feature","proposed",4*actual_width)
            logits, states = verify_tree(pending_f,p1,features,p2s,second_features,p3)
            if int(cache.get_pos()) != before:
                raise RuntimeError("tree verification moved the cache before acceptance")

            choice=univer_select(logits,p1,p2,p3,candidate_ids,candidate_q,
                second_ids,second_q,rng,policy,decide,shared.draw)
            accepted=choice['accepted_length']
            selected=choice['tokens']
            bonus=choice['bonus']
            path=choice['path']
            stats.setdefault('univer_allocation_nodes',0)
            stats['univer_allocation_nodes']+=choice['allocation_nodes']
            count('position','verified',1+(accepted>=2)+(can_long and accepted>=4))
            count('position','accepted',(accepted+1)//2)
            if accepted>=1:
                stats['feature_attempts']+=1
                count('feature','verified')
            if accepted>=2:
                branch=(path[2]-2)//(10 if can_long else 2)
                stats['feature_branch_hits']+=1
                count('feature','accepted')
                random_first=branch==len(candidate_ids)-1
                stats['random_proposal_accepts']+=int(random_first)
                stats['residual_set_hits']+=int(not random_first)
                key=str(branch)
                stats['feature_branch_index_counts'][key]=stats['feature_branch_index_counts'].get(key,0)+1
            elif accepted==1:
                stats['feature_all_rejected']+=1
            if can_long and accepted>=3:
                stats['feature2_attempts']+=1
                count('feature','verified')
            if can_long and accepted>=4:
                second_branch=(path[4]-(2+10*branch)-2)//2
                count('feature','accepted')
                stats['feature2_hits']+=1
                stats['feature2_random_accepts']+=int(second_branch==3)
                stats['feature2_residual_hits']+=int(second_branch!=3)
            commit_path(before,path)
            for offset,token in enumerate(selected):
                if offset%2==0:
                    validate_position(previous,token)
                    previous=token
                else:
                    require(P_VOCAB<=token<EOS,'Accepted Feature must be legal')
                prefix.append(token)
            if accepted>=4:
                online=shadow2
                if accepted>=5:online.finalize(p3)
            elif accepted>=2:
                online=shadow
                if accepted>=3:online.finalize(p2)
            elif accepted>=1:
                online=shadow
            prefix.append(bonus)
            if accepted%2==0:
                if bonus==EOS:
                    online.drop_eos()
                    finish_round(begin,accepted)
                    break
                validate_position(previous,bonus)
                online.finalize(bonus)
                recovery_logits,recovery_states=target_forward([bonus],recovery=True)
                next_feature=draw(recovery_logits[-1])
                require(P_VOCAB<=next_feature<EOS,'Recovery Feature must be legal')
                prefix.append(next_feature)
                h_prev=recovery_states[-1].detach().clone()
            else:
                require(P_VOCAB<=bonus<EOS,'Bonus Feature must be legal')
                h_prev=states[path[-1]].detach().clone()
            finish_round(begin,accepted)

    finally:
        stats["draft_calls"] = int(getattr(adapter, "forward_calls", 0))
        stats["draft_input_tokens"] = int(getattr(adapter, "input_tokens", 0))
        try:
            adapter.close()
        finally:
            if capture is not None:
                capture.close()
            cache = None

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    suffix = prefix[len(prompt):]
    shared.validate_sequence(prompt, suffix, ranks)
    require(len(suffix) <= budget, "Output exceeds the requested token budget")
    require(len(stats["accepted_lengths"]) == stats["rounds"]
            == len(stats["round_generated_lengths"]), "Round totals differ")
    for field in ("proposed", "verified", "accepted"):
        require(stats[field] == sum(stats[kind + "_" + field]
                                    for kind in shared.KINDS),
                "Typed counters differ")

    stats.update(
        generated_tokens=len(suffix),
        seconds=elapsed,
        tokens_per_second=ratio(len(suffix), elapsed),
        eos=bool(suffix and suffix[-1] == EOS),
        k=2,
        proposal_width=2 + width,
        tree_width=width,
        seed=int(seed),
        prompt_tokens=len(prompt) - 1,
        requested_tokens=budget,
        temperature=0.9,
        target_top_p=0.9,
        draft_top_p=float(getattr(adapter.feature, '_pfspec_draft_top_p', .9)),
        variant=f"ranker_univer_conditional_feature_set_w{width}",
        mean_accepted_length=ratio(sum(stats["accepted_lengths"]), stats["rounds"]),
        mean_generated_per_round=ratio(sum(stats["round_generated_lengths"]),
                                       stats["rounds"]),
    )
    if hasattr(tree_runtime, "stats"):
        stats["tree_runtime_stats"] = tree_runtime.stats()
    return shared._add_rates(stats), suffix


def decode(
    target,
    adapter,
    ranker,
    ranks,
    prompt,
    budget,
    seed,
    *,
    tree_runtime,
    width=2,
    cache_factory=None,
):
    """Run the decoder in inference mode with autocast disabled."""
    import torch

    device = next(target.parameters()).device
    autocast = (torch.autocast(device_type=device.type, enabled=False)
                if device.type == "cuda" else nullcontext())
    with torch.inference_mode(), autocast:
        return _decode_impl(
            target, adapter, ranker, ranks, prompt, budget, seed,
            tree_runtime=tree_runtime,
            width=width,
            cache_factory=cache_factory,
        )
