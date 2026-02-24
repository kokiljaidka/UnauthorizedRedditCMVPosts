# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 22:56:31 2026

@author: cnmkj
"""

# This is a drop-in modification of the earlier script:
# - New annotation task: Alignment Moves + Authority Moves
# - Uses your TSV "body" column (comment text)
# - Adds new cols: alignment_* and authority_* (+ counts + spans + JSON payload)

# test_sarcasm_annotation_1hour.py
import os
import json
import pandas as pd
from groq import Groq
import concurrent.futures
from typing import Dict
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import math
import numpy


import os
import re
import csv
import json
import time
import math
import concurrent.futures
from typing import Dict, Any, List

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# from groq import Groq
# from your_client_file import get_groq_client



# Initialize Groq client
def get_groq_client() -> Groq:
    """Initialize Groq client with API key from environment."""
    api_key = ""
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY environment variable not set. "
            "Please set it with: export GROQ_API_KEY='your-key-here'"
        )
    return Groq(api_key=api_key)

# System prompt for sarcasm detection
# -*- coding: utf-8 -*-
"""
Identity coding for CMV-style comments (targeting + adoption)

NOTE: Do NOT hardcode API keys in source code.
Set env var instead:
  Windows (PowerShell):  setx GROQ_API_KEY "gsk_..."
  macOS/Linux:           export GROQ_API_KEY="gsk_..."
"""

SYSTEM_PROMPT = """
You are a careful research assistant for content analysis of online comments.

Task:
Analyze ONE comment and annotate rhetorical moves in two categories:

(1) Alignment Moves:
Alignment moves capture how the speaker positioned themselves relative to the interlocutor's stance.
- POSITIVE alignment: concession, partial agreement, acknowledgment of reasonable concerns, cooperative framing.
- NEGATIVE alignment: disagreement, correction, refutation, premise challenge, or reframing of the interlocutor's assumptions.
For each instance, record:
- polarity: POSITIVE or NEGATIVE
- move_type: one of {CONCESSION, PARTIAL_AGREEMENT, ACKNOWLEDGMENT, COOPERATIVE_FRAME, DISAGREEMENT, CORRECTION, REFUTATION, PREMISE_CHALLENGE, REFRAME, OTHER}
- notes: brief justification

(2) Authority Moves:
Authority moves signal epistemic credibility. Classify each authority claim into:
- CREDENTIALS: formal education, training, or professional expertise
- EXPERIENTIAL: first-person claims grounded in direct personal experience
- INSTITUTIONAL: organizational position or governing authority
- FORUM: platform norms or procedural standards
- EXTERNAL: outside sources (laws, research, reports, published materials)
- SOCIAL_EXPECTATIONS: appeals to perceived beliefs or norms of broader social groups
For each instance, record:
- authority_type: one of {CREDENTIALS, EXPERIENTIAL, INSTITUTIONAL, FORUM, EXTERNAL, SOCIAL_EXPECTATIONS, OTHER}
- notes: brief justification

STRICT OUTPUT FORMAT:
Return ONLY valid JSON. No extra keys. No markdown.

{
  "alignment_present": true/false,
  "alignment_instances": [
    {
      "span": "... exact quoted substring from the comment ...",
      "polarity": "POSITIVE|NEGATIVE",
      "move_type": "CONCESSION|PARTIAL_AGREEMENT|ACKNOWLEDGMENT|COOPERATIVE_FRAME|DISAGREEMENT|CORRECTION|REFUTATION|PREMISE_CHALLENGE|REFRAME|OTHER",
      "notes": "short justification"
    }
  ],
  "authority_present": true/false,
  "authority_instances": [
    {
      "span": "... exact quoted substring from the comment ...",
      "authority_type": "CREDENTIALS|EXPERIENTIAL|INSTITUTIONAL|FORUM|EXTERNAL|SOCIAL_EXPECTATIONS|OTHER",
      "notes": "short justification"
    }
  ]
}

Rules:
- If no instances, return empty lists and *_present = false.
- "span" must be copied verbatim from the comment text.
- Prefer shorter spans that precisely locate the move (not the whole comment).
""".strip()


# -----------------------------
# Helpers
# -----------------------------
class RateLimitError(Exception):
    pass


def normalize_text(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    s = str(x)
    s = s.replace("\r", "\n").replace("\t", " ")
    s = "".join(ch for ch in s if (ord(ch) >= 32) or (ch == "\n"))
    s = "\n".join([ln.strip() for ln in s.split("\n")]).strip()
    return s


def strict_empty_response() -> Dict[str, Any]:
    return {
        "alignment_present": False,
        "alignment_instances": [],
        "authority_present": False,
        "authority_instances": [],
    }


def _validate_json(obj: Dict[str, Any]) -> Dict[str, Any]:
    required = ["alignment_present", "alignment_instances", "authority_present", "authority_instances"]
    for k in required:
        if k not in obj:
            raise ValueError(f"Missing key: {k}")

    if not isinstance(obj["alignment_present"], bool):
        raise ValueError("alignment_present must be boolean")
    if not isinstance(obj["authority_present"], bool):
        raise ValueError("authority_present must be boolean")
    if not isinstance(obj["alignment_instances"], list):
        raise ValueError("alignment_instances must be list")
    if not isinstance(obj["authority_instances"], list):
        raise ValueError("authority_instances must be list")

    for inst in obj["alignment_instances"]:
        if not isinstance(inst, dict):
            raise ValueError("Each alignment_instance must be an object")
        for kk in ["span", "polarity", "move_type", "notes"]:
            if kk not in inst:
                raise ValueError(f"Missing key in alignment_instances: {kk}")

    for inst in obj["authority_instances"]:
        if not isinstance(inst, dict):
            raise ValueError("Each authority_instance must be an object")
        for kk in ["span", "authority_type", "notes"]:
            if kk not in inst:
                raise ValueError(f"Missing key in authority_instances: {kk}")

    return obj


def safe_count(lst) -> int:
    return int(len(lst)) if isinstance(lst, list) else 0


def join_spans(instances: List[Dict[str, Any]], key: str = "span", sep: str = " ||| ") -> str:
    spans = []
    for inst in instances or []:
        if isinstance(inst, dict):
            s = str(inst.get(key, "")).strip()
            if s:
                spans.append(s)
    return sep.join(spans)


# -----------------------------
# API call (row-level)
# -----------------------------
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(RateLimitError),
)
def process_row(comment_text: str, client) -> Dict[str, Any]:
    try:
        clean_text = normalize_text(comment_text)
        if not clean_text or len(clean_text) < 3:
            return strict_empty_response()

        MAX_CHARS = 6000
        if len(clean_text) > MAX_CHARS:
            clean_text = clean_text[:MAX_CHARS]

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": clean_text},
            ],
            max_tokens=800,
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        result = json.loads(completion.choices[0].message.content)
        result = _validate_json(result)
        return result

    except Exception as e:
        error_str = str(e)
        if "rate_limit_exceeded" in error_str or "429" in error_str:
            raise RateLimitError(error_str)
        return strict_empty_response()


# -----------------------------
# Batch
# -----------------------------
def process_batch(texts: pd.Series, client, max_workers: int = 2) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = [None] * len(texts)  # type: ignore
    completed = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(process_row, t, client): i for i, t in enumerate(texts)}

        for future in concurrent.futures.as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                results[idx] = future.result()
            except Exception:
                results[idx] = strict_empty_response()

            completed += 1
            if completed % 25 == 0:
                print(f"Progress: {completed}/{len(texts)} ({completed/len(texts)*100:.1f}%)")

    return results


# -----------------------------
# Main
# -----------------------------
def main():
    print("=" * 70)
    print("ALIGNMENT + AUTHORITY CODING RUN")
    print("Model: llama-3.3-70b-versatile")
    print("Workers: 2")
    print("=" * 70)

    try:
        client = get_groq_client()
        print("✓ Groq client initialized")
    except ValueError as e:
        print(f"✗ {e}")
        return

    input_file = r"data/CMV-LLM-Posts_parsed.tsv"
    try:
        df = pd.read_csv(input_file, sep="\t", encoding="utf-8", on_bad_lines="skip", quoting=csv.QUOTE_MINIMAL)
        print(f"✓ Loaded {len(df)} rows")
    except Exception as e:
        print(f"✗ Failed to load TSV: {e}")
        return

    TEXT_COL = "body"  # <-- your comment text column
    if TEXT_COL not in df.columns:
        print(f"✗ Error: '{TEXT_COL}' column not found")
        print(f"Available columns: {list(df.columns)}")
        return

    # Optional sampling
    TARGET_SAMPLE = 100000
    if len(df) <= TARGET_SAMPLE:
        df_sample = df.copy()
    else:
        # stratify by author if present, else random
        STRATA_COL = "author"
        if STRATA_COL in df.columns:
            group_sizes = df.groupby(STRATA_COL).size().reset_index(name="count")
            group_sizes["sample_n"] = (group_sizes["count"] / len(df) * TARGET_SAMPLE).round().astype(int)
            diff = TARGET_SAMPLE - int(group_sizes["sample_n"].sum())
            if diff != 0:
                adj_idx = group_sizes["count"].idxmax()
                group_sizes.at[adj_idx, "sample_n"] += diff

            sample_map = dict(zip(group_sizes[STRATA_COL], group_sizes["sample_n"]))

            def _sample_group(g):
                n = int(sample_map.get(g.name, 0))
                if n <= 0:
                    return g.iloc[0:0]
                n = min(n, len(g))
                return g.sample(n=n, replace=False, random_state=42)

            df_sample = df.groupby(STRATA_COL, group_keys=False).apply(_sample_group).reset_index(drop=True)
        else:
            df_sample = df.sample(n=TARGET_SAMPLE, replace=False, random_state=42).reset_index(drop=True)

    print(f"✓ Sample size: {len(df_sample)}")

    # Normalize
    df_sample[TEXT_COL] = df_sample[TEXT_COL].apply(normalize_text)

    start = time.time()
    results = process_batch(df_sample[TEXT_COL], client, max_workers=2)

    # -----------------------------
    # New columns
    # -----------------------------
    df_sample["alignment_present"] = [r["alignment_present"] for r in results]
    df_sample["authority_present"] = [r["authority_present"] for r in results]

    df_sample["alignment_n"] = [safe_count(r.get("alignment_instances")) for r in results]
    df_sample["authority_n"] = [safe_count(r.get("authority_instances")) for r in results]

    # Polarity counts (useful for your “positive vs negative alignment frequency”)
    df_sample["alignment_pos_n"] = [
        sum(1 for inst in (r.get("alignment_instances") or []) if str(inst.get("polarity", "")).upper() == "POSITIVE")
        for r in results
    ]
    df_sample["alignment_neg_n"] = [
        sum(1 for inst in (r.get("alignment_instances") or []) if str(inst.get("polarity", "")).upper() == "NEGATIVE")
        for r in results
    ]

    # Authority-type counts (wide-ish but compact)
    AUTH_TYPES = ["CREDENTIALS", "EXPERIENTIAL", "INSTITUTIONAL", "FORUM", "EXTERNAL", "SOCIAL_EXPECTATIONS", "OTHER"]
    for t in AUTH_TYPES:
        df_sample[f"authority_{t.lower()}_n"] = [
            sum(1 for inst in (r.get("authority_instances") or []) if str(inst.get("authority_type", "")).upper() == t)
            for r in results
        ]

    # Store full instances
    df_sample["alignment_instances_json"] = [
        json.dumps(r.get("alignment_instances", []), ensure_ascii=False) for r in results
    ]
    df_sample["authority_instances_json"] = [
        json.dumps(r.get("authority_instances", []), ensure_ascii=False) for r in results
    ]

    # Span summaries for quick QA
    df_sample["alignment_spans"] = [join_spans(r.get("alignment_instances", []), key="span") for r in results]
    df_sample["authority_spans"] = [join_spans(r.get("authority_instances", []), key="span") for r in results]

    # Metadata/provenance
    df_sample["rhetoric_model"] = "llama-3.3-70b-versatile"
    df_sample["rhetoric_prompt_version"] = "v1_alignment_authority_strict"
    df_sample["rhetoric_max_chars"] = 6000

    output_file = r"data/comments_alignment_authority.tsv"
    df_sample.to_csv(output_file, sep="\t", index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)

    elapsed = time.time() - start
    print("=" * 70)
    print(f"✓ Saved: {output_file}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    if elapsed > 0:
        print(f"Rate: {len(df_sample)/(elapsed/60):.1f} rows/min")
    print("=" * 70)


if __name__ == "__main__":
    main()