"""Capacity-flood suite runner. Runs ON the GPU instance (or --tiny on CPU).

For every item in items.jsonl: one forward pass, J-lens readout over the
workspace band at the last N prompt positions, a greedy continuation for
correctness, and a row of metrics to results.jsonl. analyze.py aggregates.

    python run.py --items items.jsonl --out results.jsonl
    python run.py --tiny --items items.jsonl --out /tmp/smoke.jsonl   # CPU smoke test

Stop the dashboard first: two copies of the model do not fit a 48GB card.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

NUMBER_WORDS = set("zero one two three four five six seven eight nine ten eleven twelve".split())
NUMBERS = NUMBER_WORDS | {str(i) for i in range(13)}


def token_class(family: str):
    if family in ("arith", "count-letter"):
        return lambda s: s in NUMBERS
    if family == "nth-letter":
        return lambda s: len(s) == 1 and s.isalpha()
    return None


def single_token_ids(model, forms: list[str]) -> list[int]:
    """Ids of every surface form (with and without a leading space) that the
    tokenizer encodes as exactly one token. The lens can only name these."""
    bos = getattr(model.tokenizer, "bos_token_id", None)
    ids = []
    for form in forms:
        for text in (form, " " + form):
            enc = model.encode(text)[0].tolist()
            if bos is not None and enc and enc[0] == bos:
                enc = enc[1:]
            if len(enc) == 1:
                ids.append(enc[0])
    return sorted(set(ids))


def ranks_of(logits: torch.Tensor, ids: list[int]) -> torch.Tensor:
    """Full-vocab rank (0 = top) of each id in a [vocab] logit vector."""
    return torch.stack([(logits > logits[i]).sum() for i in ids])


def wordlike_mask(model, vocab: int) -> torch.Tensor:
    from jlens.vis import _meaningful_token_mask

    return _meaningful_token_mask(model.tokenizer, vocab, torch.device("cpu"))


@torch.no_grad()
def greedy(model, ids: torch.Tensor, n: int) -> list[int]:
    """Greedy continuation through the LensModel interface (no KV cache; n is
    small). Reads the final block's residual so unembed's norm applies once."""
    from jlens.hooks import ActivationRecorder

    final = model.n_layers - 1
    out: list[int] = []
    for _ in range(n):
        with ActivationRecorder(model.layers, at=[final]) as rec:
            model.forward(ids)
            h = rec.activations[final][0, -1].float()
        nxt = int(model.unembed(h).argmax())
        out.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
    return out


def excess_kurtosis(x: torch.Tensor) -> float:
    x = x - x.mean()
    return float((x.pow(4).mean() / x.pow(2).mean().pow(2)) - 3)


def entropy(logits: torch.Tensor) -> float:
    p = torch.log_softmax(logits, -1)
    return float(-(p.exp() * p).sum())


@torch.no_grad()
def measure(model, lens, item: dict, band: list[int], n_pos: int, gen: int, mask, top_k: int = 10) -> dict:
    prompt = item["prompt"]
    positions = list(range(-n_pos, 0))
    lens_logits, model_logits, input_ids = lens.apply(model, prompt, layers=band, positions=positions)

    ans_ids = single_token_ids(model, item["answer"])
    inter_ids = [single_token_ids(model, group) for group in item["intermediates"]]
    cls = token_class(item["family"])
    decode = lambda t: model.tokenizer.decode([int(t)]).strip()  # noqa: E731

    per_pos = []
    for pi in range(n_pos):
        rows = []  # per band layer
        top1 = []
        cands: set[int] = set()
        class_hits, class_total = 0, 0
        class_seen: set[str] = set()
        for layer in band:
            lg = lens_logits[layer][pi]
            masked = lg.masked_fill(~mask, float("-inf")) if mask is not None else lg
            tk = masked.topk(top_k).indices.tolist()
            top1.append(tk[0])
            cands.update(tk)
            if cls is not None:
                for t in tk:
                    s = decode(t)
                    class_total += 1
                    if cls(s):
                        class_hits += 1
                        class_seen.add(s)
            row = dict(
                layer=layer,
                entropy=entropy(lg),
                kurtosis=excess_kurtosis(lg),
                top1=decode(tk[0]),
                answer_rank=int(ranks_of(lg, ans_ids).min()) if ans_ids else None,
                answer_p=float(torch.softmax(lg, -1)[ans_ids].sum()) if ans_ids else None,
                inter_rank=[int(ranks_of(lg, g).min()) if g else None for g in inter_ids],
            )
            rows.append(row)
        stab = sum(a == b for a, b in zip(top1, top1[1:])) / max(1, len(top1) - 1)
        ml = model_logits[pi]
        per_pos.append(
            dict(
                position=positions[pi],
                token=decode(input_ids[0, positions[pi]]),
                layers=rows,
                band_min_answer_rank=min((r["answer_rank"] for r in rows), default=None) if ans_ids else None,
                band_min_inter_rank=[
                    min(r["inter_rank"][g] for r in rows) if inter_ids[g] else None for g in range(len(inter_ids))
                ],
                band_min_entropy=min(r["entropy"] for r in rows),
                band_max_kurtosis=max(r["kurtosis"] for r in rows),
                occupancy=len(cands),
                top1_stability=stab,
                class_share=(class_hits / class_total) if class_total else None,
                class_distinct=len(class_seen) if cls is not None else None,
                model_answer_rank=int(ranks_of(ml, ans_ids).min()) if ans_ids else None,
                model_answer_p=float(torch.softmax(ml, -1)[ans_ids].sum()) if ans_ids else None,
                model_top1=decode(ml.argmax()),
                model_top1_p=float(torch.softmax(ml, -1).max()),
            )
        )

    cont_ids = greedy(model, input_ids, gen)
    cont = model.tokenizer.decode(cont_ids)
    norm = cont.strip().strip('"\'').lower()
    correct = any(norm.startswith(a.lower()) for a in item["answer"])
    # Whole-word match: a single-letter answer would otherwise match almost anything.
    later = not correct and any(re.search(rf"\b{re.escape(a)}\b", cont, re.I) for a in item["answer"])
    return dict(
        id=item["id"],
        family=item["family"],
        tier=item["tier"],
        prompt=prompt,
        n_tokens=int(input_ids.shape[1]),
        answer=item["answer"],
        answer_single_token=bool(ans_ids),
        band_scored=bool(item.get("band_scored", True)),
        continuation=cont,
        correct=correct,
        answer_later=later,
        positions=per_pos,
    )


def load_tiny():
    sys.path.insert(0, os.path.join(REPO, "tests"))
    from tiny import TinyDecoder

    from jlens.fitting import fit

    model = TinyDecoder(n_layers=4, d_model=8)
    lens = fit(model, ["abcdefghij " * 5, "klmnopqrst " * 5], source_layers=[0, 1, 2], dim_batch=4, max_seq_len=64)
    return model, lens


def load_real():
    sys.path.insert(0, os.path.join(REPO, "deploy"))
    import dashboard

    dashboard._load()
    if dashboard._state["status"] != "ready":
        raise SystemExit(dashboard._state["error"])
    return dashboard._model, dashboard._lens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=os.path.join(HERE, "items.jsonl"))
    ap.add_argument("--out", default=os.path.join(HERE, "results.jsonl"))
    ap.add_argument("--band", nargs=2, type=int, default=[24, 58], metavar=("LO", "HI"),
                    help="workspace band, inclusive; paper's L38-92 of 100 on 64 layers")
    ap.add_argument("--positions", type=int, default=6, help="read out the last N prompt positions")
    ap.add_argument("--gen", type=int, default=12, help="greedy tokens for correctness")
    ap.add_argument("--no-mask", action="store_true", help="do not restrict top-k to word-like tokens")
    ap.add_argument("--tiny", action="store_true", help="CPU smoke test on tests/tiny.py")
    a = ap.parse_args()

    model, lens = load_tiny() if a.tiny else load_real()
    band = [l for l in lens.source_layers if a.band[0] <= l <= a.band[1]]
    if not band:
        raise SystemExit(f"no fitted layers in band {a.band}; fitted: {lens.source_layers}")
    vocab = int(model.unembed(torch.zeros(model.d_model, device=model.input_device)).shape[-1])
    mask = None if a.no_mask else wordlike_mask(model, vocab)
    items = [json.loads(l) for l in open(a.items) if l.strip()]
    print(f"{len(items)} items | band {band[0]}..{band[-1]} ({len(band)} layers) | positions {a.positions}", flush=True)

    t0 = time.time()
    with open(a.out, "w") as f:
        for n, item in enumerate(items, 1):
            r = measure(model, lens, item, band, a.positions, a.gen, mask)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            p = r["positions"][-1]
            print(
                f"[{n:3}/{len(items)}] {r['id']:<18} ok={int(r['correct'])} later={int(r['answer_later'])} "
                f"ans_rank={p['band_min_answer_rank']} inter={p['band_min_inter_rank']} "
                f"H={p['band_min_entropy']:.2f} "
                f"occ={p['occupancy']} stab={p['top1_stability']:.2f} -> {r['continuation']!r}",
                flush=True,
            )
    print(f"done in {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
