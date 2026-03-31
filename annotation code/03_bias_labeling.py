# -*- coding: utf-8 -*-
"""
Cognitive Bias / Heuristic Annotation — Production Script
Model : llama-3.3-70b-versatile  (via Groq)
Input : TSV with a "body" column
Output: TSV with per-bias count columns + span JSON

Usage
-----
  export GROQ_API_KEY="gsk_..."        # set ONCE in shell, never in code
  python 03_bias_labeling_fixed.py
"""

import csv
import json
import math
import os
import time
import logging
import concurrent.futures
from typing import Any, Dict, List, Optional

import pandas as pd
from groq import Groq
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
MODEL          = "llama-3.3-70b-versatile"
MAX_WORKERS    = 2          # Groq free-tier rate limit
MAX_CHARS      = 6_000      # hard truncation per comment
TARGET_SAMPLE  = 100_000
INPUT_FILE     = "data/CMV-LLM-Posts_parsed.tsv"
OUTPUT_FILE    = "data/comments_biases.tsv"
TEXT_COL       = "body"
STRATA_COL     = "author"   # used for stratified sampling if present
PROMPT_VERSION = "v2_biases_kahneman_fewshot"

BIAS_TYPES = [
    "LAW_OF_SMALL_NUMBERS",
    "AVAILABILITY_HEURISTIC",
    "REPRESENTATIVENESS_HEURISTIC",
    "BASE_RATE_NEGLECT",
    "ATTRIBUTE_SUBSTITUTION",
    "AFFECT_HEURISTIC",
    "CONFIRMATION_BIAS",
    "ILLUSION_OF_VALIDITY",
]


# ─────────────────────────────────────────────────────────────────────────────
# System prompt  (definitions + global rules + few-shot examples)
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = r"""
You are a careful research assistant for content analysis of online comments.

══════════════════════════════════════════════════════
GLOBAL RULE — READ BEFORE EVERYTHING ELSE
══════════════════════════════════════════════════════
The bias MUST be operative in the AUTHOR'S OWN reasoning.
• A comment that describes, quotes, diagnoses, or argues AGAINST a bias
  in someone else does NOT exhibit that bias. Do NOT annotate it.
• Explicitly hedged claims ("I guess", "maybe", "in my view", "probably")
  do NOT qualify as ILLUSION_OF_VALIDITY — unwarranted certainty requires
  the author to assert without hedging.
• Citing a well-documented historical fact confidently is NOT a bias.
• An author who accurately acknowledges both sides is NOT exhibiting
  CONFIRMATION_BIAS — that requires demonstrably suppressing counterevidence.

══════════════════════════════════════════════════════
BIAS DEFINITIONS  (Kahneman / dual-process framework)
══════════════════════════════════════════════════════

1) LAW_OF_SMALL_NUMBERS
   Inference from a limited number of observations (anecdotes, personal
   experience, a handful of events) to population-level conclusions.
   KEY: the anecdote must be GENERALISED to a broader claim. A story that
   stays personal ("I did X") does not qualify; it qualifies when followed
   by a claim like "so that's how it works for everyone."
   NEGATIVE: citing peer-reviewed aggregated statistics is NOT this bias.

2) AVAILABILITY_HEURISTIC
   Frequency or risk judged by ease of mental recall. Vivid, emotionally
   salient, or recent individual cases used as IMPLICIT EVIDENCE OF
   PREVALENCE. KEY: the case must support a claim about how common/likely
   something is — not merely establish credentials or emotional tone.
   NEGATIVE: personal expertise stated without a prevalence claim is NOT
   this bias.

3) REPRESENTATIVENESS_HEURISTIC
   Probability or likelihood assessed via similarity to a prototype or
   stereotype rather than statistical base rates. Narrative coherence or
   descriptive resemblance used instead of statistical reasoning.
   NEGATIVE: a factual comparison between two well-documented entities
   is not this bias.

4) BASE_RATE_NEGLECT
   Known population frequencies omitted or discounted in favour of
   case-based reasoning. The author ignores or dismisses statistical
   prevalence data when evaluating an individual case.
   NEGATIVE: if no base-rate information exists, there is nothing to neglect.

5) ATTRIBUTE_SUBSTITUTION
   A hard target question is replaced by a simpler evaluative proxy.
   BOTH must be identifiable: (a) the complex question being avoided AND
   (b) the simpler attribute (plausibility, moral clarity, narrative
   coherence) that substitutes for it.
   NEGATIVE: general oversimplification without an identifiable substituted
   attribute does NOT qualify. Topic drift / changing the subject is NOT
   attribute substitution.

6) AFFECT_HEURISTIC
   Immediate emotional response — moral resonance, disgust, pride, fear —
   used as implicit justification for a claim. The emotional framing must
   REPLACE rather than merely accompany analytical reasoning.
   NEGATIVE: emotionally charged language that accompanies a sound argument
   is rhetoric, not the affect heuristic.

7) CONFIRMATION_BIAS
   Selective reinforcement of pre-existing beliefs while minimising or
   excluding available disconfirming evidence. The author must demonstrably
   suppress or dismiss counterevidence, not merely argue one side.
   NEGATIVE: making a one-sided argument without acknowledging alternatives
   is advocacy, not confirmation bias.

8) ILLUSION_OF_VALIDITY
   Overconfidence in conclusions derived from internally coherent narratives
   absent sufficient evidentiary support. Forward-looking or causal claims
   presented with unwarranted certainty.
   NEGATIVE: hedged claims, historical facts, and rebuttals of someone
   else's overconfident assertion do NOT qualify.

══════════════════════════════════════════════════════
FEW-SHOT EXAMPLES
══════════════════════════════════════════════════════

── POSITIVE EXAMPLE 1 (LAW_OF_SMALL_NUMBERS + AVAILABILITY_HEURISTIC) ──
Comment:
  "Dating apps haven't made people less approachable. I've seen this
  firsthand in LA, especially in places like Silver Lake and Echo Park
  where community spaces and social events are thriving. Look at how
  farmers markets, community gardens, and local activism groups have
  exploded in popularity."
Output:
{
  "bias_present": true,
  "bias_instances": [
    {
      "span": "I've seen this firsthand in LA, especially in places like Silver Lake and Echo Park",
      "bias_type": "LAW_OF_SMALL_NUMBERS",
      "notes": "Author generalises a societal claim from personal observation in one city neighbourhood."
    },
    {
      "span": "Look at how farmers markets, community gardens, and local activism groups have exploded in popularity",
      "bias_type": "AVAILABILITY_HEURISTIC",
      "notes": "Vivid, easily recalled examples used as implicit evidence that society is more connected overall."
    }
  ]
}

── POSITIVE EXAMPLE 2 (ILLUSION_OF_VALIDITY + CONFIRMATION_BIAS) ──
Comment:
  "The core problem with your argument is that you're defining conservatism
  only by its losses, not its victories. Without conservatives, we'd
  implement every new idea without testing, leading to disasters. Society
  needs both change AND resistance to change — this is just how progress
  works."
Output:
{
  "bias_present": true,
  "bias_instances": [
    {
      "span": "Without conservatives, we'd implement every new idea without testing, leading to disasters",
      "bias_type": "ILLUSION_OF_VALIDITY",
      "notes": "Causal claim about a counterfactual asserted with unwarranted certainty and no supporting evidence."
    },
    {
      "span": "you're defining conservatism only by its losses, not its victories",
      "bias_type": "CONFIRMATION_BIAS",
      "notes": "Author selectively frames the opponent's evidence while presenting their own narrative as balanced."
    }
  ]
}

── POSITIVE EXAMPLE 3 (ATTRIBUTE_SUBSTITUTION) ──
Comment:
  "NYC hotels are incredibly overpriced compared to other big cities.
  It's a function of insane real estate prices and all-around financial
  shenanigans — that's just what happens when greed takes over."
Output:
{
  "bias_present": true,
  "bias_instances": [
    {
      "span": "It's a function of insane real estate prices and all-around financial shenanigans",
      "bias_type": "ATTRIBUTE_SUBSTITUTION",
      "notes": "The complex economic question of hotel pricing is replaced by a simpler moral verdict ('greed/shenanigans') without evidentiary support."
    }
  ]
}

── POSITIVE EXAMPLE 4 (CONFIRMATION_BIAS) ──
Comment:
  "The real problem isn't high-paying jobs — it's unregulated capitalism
  and weak labor laws. That's the only explanation for why the system
  keeps failing working people."
Output:
{
  "bias_present": true,
  "bias_instances": [
    {
      "span": "That's the only explanation for why the system keeps failing working people",
      "bias_type": "CONFIRMATION_BIAS",
      "notes": "Author asserts a single ideologically congruent explanation and excludes all alternative accounts."
    }
  ]
}

── POSITIVE EXAMPLE 5 (BASE_RATE_NEGLECT) ──
Comment:
  "Black Americans are still 3x more likely to be killed by police than
  white Americans, yet you keep focusing on these individual cases as if
  they're outliers."
Output:
{
  "bias_present": true,
  "bias_instances": [
    {
      "span": "you keep focusing on these individual cases as if they're outliers",
      "bias_type": "BASE_RATE_NEGLECT",
      "notes": "Author correctly notes the opponent is discounting known population-level statistics in favour of case-by-case reasoning."
    }
  ]
}

── NEGATIVE EXAMPLE 1 (bias_present = false) ──
Comment:
  "The behavior you're describing isn't really about being a furry — it's
  about being an immature teenager who doesn't understand boundaries. I'm
  part of the furry community and 99% of us cringe at that kind of
  behavior too. Think about it like anime fans: would you say 'all anime
  fans are cringe' just because a few kids at your school Naruto-run
  through the halls?"
Output:
{
  "bias_present": false,
  "bias_instances": []
}
Reason (not in output): The 99% figure is a social-norm claim, not a
frequency inference from ease of recall. The anime analogy is an explicit
reductio argument, not representativeness. The text argues analytically
throughout.

── NEGATIVE EXAMPLE 2 (bias_present = false — describes bias, does not exhibit it) ──
Comment:
  "The real issue here is confirmation bias. You're probably seeing more
  white people discuss cutting contact because they're overrepresented in
  English-speaking online spaces — that skews your sample."
Output:
{
  "bias_present": false,
  "bias_instances": []
}
Reason (not in output): The author DIAGNOSES availability/confirmation
bias in the reader's reasoning. The author's own argument is statistical
and analytical. The bias is not operative in the author's reasoning.

── NEGATIVE EXAMPLE 3 (bias_present = false — hedged claim, not illusion of validity) ──
Comment:
  "I work in healthcare economics and your cost argument seems flawed to
  me. The average NICU graduate probably contributes more to the system
  than their initial care costs, though the long-term data on this is
  still being collected."
Output:
{
  "bias_present": false,
  "bias_instances": []
}
Reason (not in output): The word "probably" and explicit acknowledgement
that data is pending are epistemic hedges. This is the opposite of
illusion of validity.

══════════════════════════════════════════════════════
OUTPUT FORMAT — STRICT JSON, NO MARKDOWN, NO EXTRA KEYS
══════════════════════════════════════════════════════
{
  "bias_present": true | false,
  "bias_instances": [
    {
      "span": "<exact substring from the comment>",
      "bias_type": "<one of the 8 types above>",
      "notes": "<one sentence justification>"
    }
  ]
}
Rules:
• If no bias, return bias_present=false and an empty list.
• Prefer the shortest span that precisely locates the cue.
• Do NOT use any bias type not listed above. No "OTHER".
• Do NOT annotate a span that is in a quote the author is rebutting.
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# Client
# ─────────────────────────────────────────────────────────────────────────────
def get_client() -> Groq:
    """Initialize Groq client with API key from environment."""
    api_key = ""
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY environment variable not set. "
            "Please set it with: export GROQ_API_KEY='your-key-here'"
        )
    return Groq(api_key=api_key)


# ─────────────────────────────────────────────────────────────────────────────
# Text helpers
# ─────────────────────────────────────────────────────────────────────────────
def normalize_text(x: Any) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    s = str(x).replace("\r", "\n").replace("\t", " ")
    s = "".join(ch for ch in s if ord(ch) >= 32 or ch == "\n")
    s = "\n".join(ln.strip() for ln in s.split("\n")).strip()
    return s


def truncate(text: str, limit: int = MAX_CHARS) -> str:
    """Truncate at last sentence boundary within limit to avoid cutting mid-argument."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    # walk back to last sentence end
    for ch in reversed(range(len(cut))):
        if cut[ch] in ".!?\n":
            return cut[: ch + 1] + " [truncated]"
    return cut + " [truncated]"


# ─────────────────────────────────────────────────────────────────────────────
# Schema validation
# ─────────────────────────────────────────────────────────────────────────────
def empty_response() -> Dict[str, Any]:
    return {"bias_present": False, "bias_instances": []}


def validate(obj: Dict[str, Any]) -> Dict[str, Any]:
    if "bias_present" not in obj or not isinstance(obj["bias_present"], bool):
        raise ValueError("bias_present missing or not bool")
    if "bias_instances" not in obj or not isinstance(obj["bias_instances"], list):
        raise ValueError("bias_instances missing or not list")
    for inst in obj["bias_instances"]:
        for k in ("span", "bias_type", "notes"):
            if k not in inst:
                raise ValueError(f"Instance missing key: {k}")
        bt = str(inst["bias_type"]).upper()
        if bt not in BIAS_TYPES:
            # coerce unknown types rather than fail silently
            log.warning("Unknown bias_type '%s' — dropped", bt)
            obj["bias_instances"].remove(inst)
    # keep bias_present consistent with list
    obj["bias_present"] = len(obj["bias_instances"]) > 0
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# API call
# ─────────────────────────────────────────────────────────────────────────────
class RateLimitError(Exception):
    pass


@retry(
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=2, min=3, max=30),
    retry=retry_if_exception_type(RateLimitError),
    reraise=True,
)
def call_api(text: str, client: Groq) -> Dict[str, Any]:
    clean = truncate(normalize_text(text))
    if len(clean) < 5:
        return empty_response()

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": clean},
        ],
        max_tokens=1_000,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content
    return validate(json.loads(raw))


def process_row(text: str, client: Groq) -> Dict[str, Any]:
    try:
        return call_api(text, client)
    except RateLimitError:
        log.error("Rate-limit retries exhausted — returning empty for this row")
        return empty_response()
    except json.JSONDecodeError as e:
        log.warning("JSON parse error: %s", e)
        return empty_response()
    except ValueError as e:
        log.warning("Schema validation error: %s", e)
        return empty_response()
    except Exception as e:
        err = str(e)
        if "rate_limit" in err.lower() or "429" in err:
            raise RateLimitError(err)
        log.warning("Unexpected error (%s): %s", type(e).__name__, err)
        return empty_response()


# ─────────────────────────────────────────────────────────────────────────────
# Batch
# ─────────────────────────────────────────────────────────────────────────────
def process_batch(
    texts: pd.Series,
    client: Groq,
    max_workers: int = MAX_WORKERS,
) -> List[Dict[str, Any]]:
    results: List[Optional[Dict[str, Any]]] = [None] * len(texts)
    completed = 0
    total = len(texts)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {pool.submit(process_row, t, client): i for i, t in enumerate(texts)}
        for fut in concurrent.futures.as_completed(future_map):
            idx = future_map[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                log.error("Row %d failed: %s", idx, e)
                results[idx] = empty_response()
            completed += 1
            if completed % 50 == 0 or completed == total:
                pct = completed / total * 100
                log.info("Progress: %d/%d  (%.1f%%)", completed, total, pct)

    return results  # type: ignore[return-value]


# ─────────────────────────────────────────────────────────────────────────────
# Sampling
# ─────────────────────────────────────────────────────────────────────────────
def stratified_sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if STRATA_COL not in df.columns:
        return df.sample(n=min(n, len(df)), random_state=42).reset_index(drop=True)

    grp = df.groupby(STRATA_COL).size().reset_index(name="cnt")
    grp["sample_n"] = (grp["cnt"] / len(df) * n).round().astype(int)
    diff = n - grp["sample_n"].sum()
    grp.at[grp["cnt"].idxmax(), "sample_n"] += diff
    size_map = dict(zip(grp[STRATA_COL], grp["sample_n"]))

    def _draw(g):
        k = max(0, min(int(size_map.get(g.name, 0)), len(g)))
        return g.sample(n=k, random_state=42) if k else g.iloc[:0]

    return (
        df.groupby(STRATA_COL, group_keys=False)
        .apply(_draw)
        .reset_index(drop=True)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Column builders
# ─────────────────────────────────────────────────────────────────────────────
def safe_count(lst: Any) -> int:
    return len(lst) if isinstance(lst, list) else 0


def join_spans(instances: List[Dict], sep: str = " ||| ") -> str:
    return sep.join(
        str(i.get("span", "")).strip()
        for i in (instances or [])
        if isinstance(i, dict) and i.get("span", "").strip()
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    log.info("=" * 65)
    log.info("BIAS / HEURISTICS CODING  |  model=%s", MODEL)
    log.info("=" * 65)

    # ── 1. Client ──────────────────────────────────────────────────────────
    try:
        client = get_client()
        log.info("Groq client initialised")
    except EnvironmentError as e:
        log.error("%s", e)
        return

    # ── 2. Load ────────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(
            INPUT_FILE, sep="\t", encoding="utf-8",
            on_bad_lines="skip", quoting=csv.QUOTE_MINIMAL,
        )
        log.info("Loaded %d rows from %s", len(df), INPUT_FILE)
    except Exception as e:
        log.error("Failed to load TSV: %s", e)
        return

    if TEXT_COL not in df.columns:
        log.error("Column '%s' not found. Available: %s", TEXT_COL, list(df.columns))
        return

    # ── 3. Sample ──────────────────────────────────────────────────────────
    if len(df) > TARGET_SAMPLE:
        df = stratified_sample(df, TARGET_SAMPLE)
        log.info("Sampled down to %d rows", len(df))

    df[TEXT_COL] = df[TEXT_COL].apply(normalize_text)

    # ── 4. Annotate ────────────────────────────────────────────────────────
    t0 = time.time()
    results = process_batch(df[TEXT_COL], client)
    elapsed = time.time() - t0

    # ── 5. Write output columns ────────────────────────────────────────────
    df["bias_present"]       = [r["bias_present"] for r in results]
    df["bias_n"]             = [safe_count(r.get("bias_instances")) for r in results]

    for bt in BIAS_TYPES:
        col = f"bias_{bt.lower()}_n"
        df[col] = [
            sum(
                1 for inst in (r.get("bias_instances") or [])
                if str(inst.get("bias_type", "")).upper() == bt
            )
            for r in results
        ]

    df["bias_instances_json"] = [
        json.dumps(r.get("bias_instances", []), ensure_ascii=False)
        for r in results
    ]
    df["bias_spans"]          = [join_spans(r.get("bias_instances", [])) for r in results]
    df["bias_model"]          = MODEL
    df["bias_prompt_version"] = PROMPT_VERSION
    df["bias_max_chars"]      = MAX_CHARS

    # ── 6. Save ────────────────────────────────────────────────────────────
    df.to_csv(OUTPUT_FILE, sep="\t", index=False,
              encoding="utf-8", quoting=csv.QUOTE_MINIMAL)

    log.info("=" * 65)
    log.info("Saved → %s", OUTPUT_FILE)
    log.info("Elapsed: %.1f min  |  Rate: %.0f rows/min",
             elapsed / 60, len(df) / (elapsed / 60) if elapsed else 0)
    log.info("bias_present=True: %d / %d  (%.1f%%)",
             df["bias_present"].sum(), len(df),
             df["bias_present"].mean() * 100)
    for bt in BIAS_TYPES:
        col = f"bias_{bt.lower()}_n"
        log.info("  %-38s  %d instances", bt, df[col].sum())
    log.info("=" * 65)


if __name__ == "__main__":
    main()
