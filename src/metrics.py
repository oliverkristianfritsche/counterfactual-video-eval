"""Scoring — ports the reference metrics for both benchmarks.

HORNet QA (reference/hornet/reward.py, util.py):
  * Open-ended QA   -> F1-Lev  (reward.string_f1 with is_simple=True)
  * Multiple-choice -> Accuracy (util.extract_one_digit_answer == gold index)
TimeBlind (reference/timeblind/utils.py): yes/no and A/B extraction, plus Q/V/I_Acc
grouping (utils.get_scores).

F1-Lev = 0.1 * token-level F1 + 0.9 * normalized Levenshtein similarity of the FIRST
tokens, computed after spacy lemmatization. The first-token behavior matches the
original (answers are prompted to be one word / answer-first) and is replicated exactly.

Dependency fallbacks (so the harness runs anywhere) are flagged, never silent:
  * `Levenshtein` missing -> pure-python DP (identical values, slower)
  * `spacy`/en_core_web_sm missing -> identity lemmatizer (values may differ from the
    paper's) — check metric_impl() in summary.json before comparing numbers.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

# ---------------------------------------------------------------- dependency shims
try:
    import Levenshtein as _lev_mod

    def _lev_distance(a: str, b: str) -> int:
        return _lev_mod.distance(a, b)

    _LEV_IMPL = "python-Levenshtein"
except ImportError:
    def _lev_distance(a: str, b: str) -> int:
        # classic DP, identical result to Levenshtein.distance
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]

    _LEV_IMPL = "pure-python (identical values)"

_NLP = None
_LEMMA_IMPL = "identity (INSTALL spacy + en_core_web_sm for paper-comparable F1-Lev)"
try:
    import spacy

    try:
        _NLP = spacy.load("en_core_web_sm")
        _LEMMA_IMPL = "spacy/en_core_web_sm (matches paper)"
    except OSError:
        _LEMMA_IMPL = "identity (spacy installed but en_core_web_sm missing: python -m spacy download en_core_web_sm)"
except ImportError:
    pass


def metric_impl() -> dict:
    """Which implementations are active — stamped into summary.json for comparability."""
    return {"levenshtein": _LEV_IMPL, "lemmatizer": _LEMMA_IMPL}


def lemmatize(text: str) -> str:
    if _NLP is None:
        return text
    return " ".join(tok.lemma_ for tok in _NLP(text))


# ---------------------------------------------------------------- F1-Lev (open-ended)
def _norm_tokens(s: str) -> list[str]:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return [t for t in s.split() if t]


def _token_f1(pred: str, gold: str) -> float:
    """reward.string_f1_simple — token-set F1."""
    pred_t, gold_t = _norm_tokens(pred), _norm_tokens(gold)
    if not gold_t or not pred_t:
        return 0.0
    inter = set(pred_t) & set(gold_t)
    if not inter:
        return 0.0
    precision = len(inter) / len(set(pred_t))
    recall = len(inter) / len(set(gold_t))
    return 2 * precision * recall / (precision + recall)


def _edit_f1(pred: str, gold: str) -> float:
    """reward.edit_f1 (is_simple=True) — Levenshtein similarity of the FIRST tokens."""
    pred_t, gold_t = _norm_tokens(pred), _norm_tokens(gold)
    if not gold_t or not pred_t:
        return 0.0
    a, b = pred_t[0], gold_t[0]
    return 1 - _lev_distance(a, b) / max(len(a), len(b))


def f1_lev(pred: str, gold: str) -> float:
    """reward.string_f1(pred, gold, is_simple=True) — the paper's open-ended metric."""
    if not pred or not str(pred).strip():
        return 0.0
    pred, gold = lemmatize(str(pred)), lemmatize(str(gold))
    return 0.1 * _token_f1(pred, gold) + 0.9 * _edit_f1(pred, gold)


# ---------------------------------------------------------------- yes/no (TimeBlind)
import unicodedata as _ud

_YESNO_PATTERNS = [  # verbatim from reference/timeblind/utils.py extract_answer
    r"(?i)(?:final(?: answer)?|answer|prediction)\s*[:：]\s*(yes|no)\b",
    r"(?i)\b(yes|no)\b(?=[\s\.\,\!\?\)]|$)",
]


def extract_yes_no(text) -> int | None:
    """Official TimeBlind extraction: 1=yes, 0=no, None=invalid (their -1)."""
    if not text or not str(text).strip():
        return None
    s = _ud.normalize("NFKC", str(text))
    s = re.sub(r"\s+", " ", s).strip()
    for pat in _YESNO_PATTERNS:
        m = re.search(pat, s)
        if m:
            return 1 if m.group(1).lower() == "yes" else 0
    return None


_AB_PATTERNS = [  # verbatim from reference/timeblind/utils.py extract_answer (multiple_choice)
    r"(?i)(?:final(?: answer)?|answer|prediction)\s*[:：]\s*([AB])\b",
    r"(?i)(?:option|choice)\s*[:：]?\s*([AB])\b",
    r"(?i)[\(\[\{]\s*([AB])\s*[\)\]\}]",
    r"(?i)\b([AB])\s*[\.\)]\b",
    r"(?i)(?<![A-Za-z0-9/])([AB])(?![A-Za-z0-9/])",
]


def extract_ab(text) -> int | None:
    """Official TimeBlind MC extraction: 1=A, 0=B, None=invalid (their -1)."""
    if not text or not str(text).strip():
        return None
    s = _ud.normalize("NFKC", str(text))
    s = re.sub(r"\s+", " ", s).strip()
    for i, pat in enumerate(_AB_PATTERNS):
        if i < len(_AB_PATTERNS) - 1:
            m = re.search(pat, s)
            if m:
                return 1 if m.group(1).upper() == "A" else 0
        else:
            # last (loosest) pattern: prefer a match near an "answer" keyword, per upstream
            kw = re.search(r"(?i)\b(final(?: answer)?|answer|prediction)\b", s)
            if kw:
                for seg in (s[:kw.start()], s[kw.end():]):
                    m = re.search(pat, seg)
                    if m:
                        return 1 if m.group(1).upper() == "A" else 0
            m = re.search(pat, s)
            if m:
                return 1 if m.group(1).upper() == "A" else 0
    return None


def score_timeblind(prediction, gold_answer, question_type):
    """Type-aware official scoring. gold: yes/no or A/B -> 1/0; invalid parse => wrong."""
    if question_type == "multiple_choice":
        parsed = extract_ab(prediction)
        gold = 1 if str(gold_answer).strip().upper() == "A" else 0
    else:
        parsed = extract_yes_no(prediction)
        gold = 1 if str(gold_answer).strip().lower() == "yes" else 0
    return (parsed is not None and parsed == gold), parsed


def score_yes_no(prediction, gold_answer):
    """Returns (correct, parsed). gold_answer is 'yes'/'no'. Invalid parse => wrong."""
    parsed = extract_yes_no(prediction)
    gold = 1 if str(gold_answer).strip().lower() == "yes" else 0
    return (parsed is not None and parsed == gold), parsed


# ---------------------------------------------------------------- first/second (contrastive)
_FIRST_SECOND_PATTERNS = [
    r"(?:final(?: answer)?|answer|prediction)\s*[:：]?\s*(first|second|1|2)\b",
    r"\b(first|second)\b",
    r"\bvideo\s*(1|2|one|two)\b",
    # bare digits only: the bare words one/two are usually pronouns ("this one"), and
    # parsing them would hide refusals from the invalid-rate gate
    r"\b(1|2)\b",
]


def extract_first_second(text) -> int | None:
    """Contrastive two-video 2AFC extraction: 1=first, 0=second, None=invalid."""
    if not text or not str(text).strip():
        return None
    s = re.sub(r"\s+", " ", _ud.normalize("NFKC", str(text)).strip().lower())
    for pat in _FIRST_SECOND_PATTERNS:
        m = re.search(pat, s)
        if m:
            return 1 if m.group(1) in ("first", "1", "one") else 0
    return None


def score_first_second(prediction, gold_answer):
    """Returns (correct, parsed). gold_answer 'first'/'second' -> 1/0. Invalid parse => wrong."""
    parsed = extract_first_second(prediction)
    gold = 1 if str(gold_answer).strip().lower() == "first" else 0
    return (parsed is not None and parsed == gold), parsed


def timeblind_pair_metrics(records: list[dict]) -> dict:
    """Q_Acc / V_Acc / I_Acc from complete 4-role groups (reference/timeblind/utils.py
    get_scores). Uses per-record `correct` + `role`; an invalid/errored answer is wrong,
    matching the official -1 handling. Incomplete groups are dropped (counted)."""
    groups: dict = defaultdict(dict)
    choices: dict = defaultdict(dict)   # parsed_choice per role, for the sticky-rate
    for r in records:
        gid, role = r.get("instance_id"), r.get("role")
        if gid is not None and role is not None:
            groups[gid][role] = bool(r.get("correct")) and not r.get("error")
            choices[gid][role] = r.get("parsed_choice")
    complete = {g: v for g, v in groups.items() if len(v) == 4}
    if not complete:
        return {}
    q = v = i = 0
    for roles in complete.values():
        q_score = (roles[0] and roles[1]) + (roles[2] and roles[3])  # per question, both videos
        v_score = (roles[0] and roles[2]) + (roles[1] and roles[3])  # per video, both questions
        q += q_score
        v += v_score
        i += 1 if (q_score == 2 and v_score == 2) else 0
    n = len(complete)
    out = {
        "q_acc": q / (n * 2),
        "v_acc": v / (n * 2),
        "i_acc": i / n,           # == the benchmark's Instance Accuracy
        "num_complete_groups": n,
    }
    if len(groups) != n:
        out["incomplete_groups_dropped"] = len(groups) - n

    # sticky-rate: fraction of question-pairs (roles 0-1, 2-3) given the SAME answer despite
    # opposite gold. 1.0 = never distinguishes a clip from its counterfactual (blind). Read
    # against sticky_prior = p^2+(1-p)^2 (the rate a video-blind model with this answer
    # marginal p would show): sticky_rate >> sticky_prior means deterministically stuck.
    def _sticky(gids) -> float:
        same = tot = 0
        for g in gids:
            ch = choices[g]
            for a, b in ((0, 1), (2, 3)):
                tot += 1
                same += ch.get(a) == ch.get(b)
        return same / tot if tot else 0.0

    def _prior(gids) -> float:
        ones = ans = 0
        for g in gids:
            for pc in choices[g].values():
                if pc is not None:
                    ans += 1
                    ones += int(pc == 1)
        p = ones / ans if ans else 0.0
        return p * p + (1 - p) * (1 - p)

    out["sticky_rate"] = _sticky(list(complete))
    out["sticky_prior"] = _prior(list(complete))

    # per-category breakdown (groups are type-homogeneous: yes_no vs multiple_choice)
    gid_cat = {}
    for r in records:
        if r.get("instance_id") in complete and r.get("category"):
            gid_cat.setdefault(r["instance_id"], r["category"])
    cats = set(gid_cat.values())
    if len(cats) > 1:
        for cat in sorted(cats):
            gids = [g for g, c in gid_cat.items() if c == cat]
            qc = vc = ic = 0
            for g in gids:
                roles = complete[g]
                qs = (roles[0] and roles[1]) + (roles[2] and roles[3])
                vs = (roles[0] and roles[2]) + (roles[1] and roles[3])
                qc += qs
                vc += vs
                ic += 1 if (qs == 2 and vs == 2) else 0
            m = len(gids)
            out[f"q_acc/{cat}"] = qc / (m * 2)
            out[f"v_acc/{cat}"] = vc / (m * 2)
            out[f"i_acc/{cat}"] = ic / m
            out[f"sticky_rate/{cat}"] = _sticky(gids)
            out[f"sticky_prior/{cat}"] = _prior(gids)
    return out


def contrastive_metrics(records: list[dict]) -> dict:
    """Instance-level summary for the two-video first/second task (the contrastive two-video task). The
    confirmatory stats (gold-slope regression, label-permutation null, gates) are computed offline
    from predictions.jsonl; this is the quick run-level read. Records carry id 'pair|pole|order',
    gold in `answer` (first/second), and parsed_choice (1=first, 0=second, None=invalid)."""
    ok = [r for r in records if not r.get("error")]
    valid = [r for r in ok if r.get("parsed_choice") is not None]
    out: dict = {}
    if valid:  # position-bias readout: 0.5 balanced, ->1 favours "first"
        out["first_rate"] = sum(1 for r in valid if r["parsed_choice"] == 1) / len(valid)

    # order-consistency and the shortcut-purged simple primary need the chosen/gold CLIP, which
    # depends on the presentation order: in AB "first"=clip A, in BA "first"=clip B.
    def _pos_clip(order, is_first):
        return "A" if (is_first == (order == "AB")) else "B"

    inst: dict = defaultdict(dict)   # instance_id (pair|pole) -> {order: record}
    for r in ok:
        parts = str(r.get("id", "")).split("|")
        if len(parts) == 3:
            inst[r.get("instance_id")][parts[2]] = r
    consistent = aligned = complete = 0
    for recs in inst.values():
        if set(recs) != {"AB", "BA"}:
            continue
        complete += 1
        pa, pb = recs["AB"].get("parsed_choice"), recs["BA"].get("parsed_choice")
        if pa is None or pb is None:
            continue
        clip_ab, clip_ba = _pos_clip("AB", pa == 1), _pos_clip("BA", pb == 1)
        if clip_ab == clip_ba:               # picked the same physical clip in both orders
            consistent += 1
            gold_clip = _pos_clip("AB", str(recs["AB"].get("answer")).lower() == "first")
            if clip_ab == gold_clip:
                aligned += 1
    if complete:
        out["order_consistency"] = consistent / complete
        # among order-consistent items, P(chosen = gold clip); shortcut floor 0.5, chance-of-motion
        out["consistent_gold_aligned"] = aligned / consistent if consistent else 0.0
        out["n_instances"] = complete
    return out


# ---------------------------------------------------------------- MCQ (choice)
def extract_one_digit_answer(text: str):
    """util.extract_one_digit_answer — first standalone digit 0-9, else None."""
    match = re.search(r"\b(\d)\b", str(text))
    return int(match.group(1)) if match else None


def extract_letter_answer(text: str, num_options: int):
    """Letter MCQ protocol: 0-based index of the first option letter (a/b/c/...) found, else None."""
    letters = [chr(ord("a") + i) for i in range(num_options)]
    for tok in _norm_tokens(str(text)):
        if tok in letters:
            return letters.index(tok)
    return None


def score_mcq(prediction, gold_index, options=None, protocol: str = "digit"):
    """Returns (correct: bool, parsed: int|None). parsed None => unparseable output."""
    if protocol == "letter":
        parsed = extract_letter_answer(prediction, len(options or []) or 26)
    else:
        parsed = extract_one_digit_answer(prediction)
    if gold_index is None or parsed is None:
        return False, parsed
    return parsed == int(gold_index), parsed


# ---------------------------------------------------------------- aggregation
def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def compute_metrics(records: list[dict], task: str) -> dict:
    """Aggregate per-sample records into summary metrics.

    Computes accuracy (question-level), instance accuracy (all questions of an
    instance/video correct), num_videos, per-category breakdowns, invalid/error rates,
    and the HORNet-paper efficiency columns (frame-selection / VLM time, avg frames).
    Sampling type and frames/fps are stamped by the runner.
    """
    n = len(records)
    ok = [r for r in records if not r.get("error")]
    metrics: dict = {"total": n, "errors": n - len(ok)}

    # instance accuracy + video count — errored questions count their instance as wrong
    groups: dict = defaultdict(list)
    for r in records:
        gid = r.get("instance_id")
        if gid is not None:
            groups[gid].append(bool(r.get("correct")) and not r.get("error"))
    if groups:
        metrics["num_videos"] = len(groups)
        metrics["instance_accuracy"] = sum(all(v) for v in groups.values()) / len(groups)

    if task in ("choice", "yesno", "contrastive"):
        correct = sum(1 for r in ok if r.get("correct"))
        invalid = sum(1 for r in ok if r.get("parsed_choice") is None)
        metrics["accuracy"] = correct / len(ok) if ok else 0.0
        metrics["correct"] = correct
        metrics["invalid_rate"] = invalid / len(ok) if ok else 0.0
        # distribution of parsed choices (surfaces positional / yes bias)
        metrics["choice_dist"] = dict(Counter(
            str(r.get("parsed_choice")) for r in ok
        ))
        if task == "yesno":
            metrics.update(timeblind_pair_metrics(records))
        elif task == "contrastive":
            metrics.update(contrastive_metrics(records))
    else:  # open-ended: msqa / general
        scores = [r.get("f1_lev", 0.0) for r in ok]
        metrics["f1_lev"] = sum(scores) / len(scores) if scores else 0.0
        metrics["exact_match_rate"] = (
            sum(1 for r in ok if r.get("correct")) / len(ok) if ok else 0.0
        )

    # per-category breakdown (accuracy for choice/yesno/contrastive, f1_lev for open-ended)
    acc_style = task in ("choice", "yesno", "contrastive")
    per_cat = defaultdict(list)
    for r in ok:
        val = float(r.get("correct", False)) if acc_style else r.get("f1_lev", 0.0)
        per_cat[r.get("category") or "all"].append(val)
    for cat, vals in sorted(per_cat.items()):
        key = "accuracy" if acc_style else "f1_lev"
        metrics[f"{key}/{cat}"] = sum(vals) / len(vals)

    # efficiency columns — matches paper Tables 2/3 (Frame Sel (s), VLM Proc (s), Avg Frames)
    for field, name in [("sel_time", "frame_sel_s"), ("vlm_time", "vlm_proc_s")]:
        vals = sorted(r[field] for r in ok if r.get(field) is not None)
        if vals:
            metrics[f"{name}/mean"] = sum(vals) / len(vals)
            metrics[f"{name}/p50"] = _percentile(vals, 0.50)
            metrics[f"{name}/p90"] = _percentile(vals, 0.90)
    frames = [r["frames_used"] for r in ok if r.get("frames_used") is not None]
    if frames:
        metrics["avg_frames"] = sum(frames) / len(frames)
    total_time = sum(
        (r.get("sel_time") or 0) + (r.get("vlm_time") or 0) for r in ok
    )
    if total_time > 0:
        metrics["throughput_samples_per_s"] = len(ok) / total_time

    return metrics
