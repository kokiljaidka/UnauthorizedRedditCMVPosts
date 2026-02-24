# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 23:00:50 2026

@author: cnmkj
"""

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


# New annotation task: Cognitive Bias / Heuristic markers (Kahneman-style)
# - Input TSV column: "body"
# - Output: bias_present + bias_instances (+ per-bias counts + spans + JSON payload)
# - Strict JSON schema from the model (no extra keys)

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


SYSTEM_PROMPT = """
You are a careful research assistant for content analysis of online comments.

Context:
Biases arise when intuitive heuristics substitute for statistical reasoning, when vivid cases override base rates,
or when coherent narratives are treated as predictive evidence. Kahneman's distinction between fast, intuitive System 1
processing and slower, deliberative System 2 reasoning suggests that fluently framed, emotionally salient, or otherwise
vivid arguments may exploit heuristic pathways rather than statistical reasoning. Confirmation bias can predispose
individuals to accept belief-consistent information while minimizing disconfirming evidence.

Task:
Analyze ONE comment and annotate instances of the following biases / heuristics:

1) LAW_OF_SMALL_NUMBERS:
Inference from a limited number of observations to population-level conclusions. Isolated anecdotes or short event
sequences treated as representative of broader distributions.

2) AVAILABILITY_HEURISTIC:
Judgment of frequency/risk based on ease of recall. Reliance on vivid, emotionally salient, or recent cases as implicit
evidence of prevalence.

3) REPRESENTATIVENESS_HEURISTIC:
Likelihood assessed via similarity to a prototype rather than statistical probability. Probability inferred from
narrative coherence, stereotype fit, or descriptive resemblance.

4) BASE_RATE_NEGLECT:
Known population frequencies are omitted/discounted in favor of case-based reasoning.

5) ATTRIBUTE_SUBSTITUTION:
Complex probabilistic judgment replaced with simpler evaluative assessment (plausibility, moral clarity, narrative
coherence) as a substitute for evidentiary sufficiency.

6) AFFECT_HEURISTIC:
Immediate emotional response used to guide evaluation. Moral resonance or emotional alignment treated as justification.

7) CONFIRMATION_BIAS:
Selective reinforcement of pre-existing beliefs while minimizing/excluding disconfirming information. Ideologically
congruent claims amplified without balanced counterevidence.

8) ILLUSION_OF_VALIDITY:
Overconfidence in conclusions drawn from internally coherent narratives absent sufficient evidentiary support.
Forward-looking or causal claims presented with unwarranted certainty.

For each instance, quote the EXACT span, label the bias type, and briefly justify.

STRICT OUTPUT FORMAT:
Return ONLY valid JSON. No extra keys. No markdown.

{
  "bias_present": true/false,
  "bias_instances": [
    {
      "span": "... exact quoted substring from the comment ...",
      "bias_type": "LAW_OF_SMALL_NUMBERS|AVAILABILITY_HEURISTIC|REPRESENTATIVENESS_HEURISTIC|BASE_RATE_NEGLECT|ATTRIBUTE_SUBSTITUTION|AFFECT_HEURISTIC|CONFIRMATION_BIAS|ILLUSION_OF_VALIDITY|OTHER",
      "notes": "short justification"
    }
  ]
}

Rules:
- If no instances, return empty list and bias_present = false.
- Prefer short spans that precisely locate the cue.
- Use OTHER only if a clear bias cue is present but none of the listed types apply.
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
    return {"bias_present": False, "bias_instances": []}


def _validate_json(obj: Dict[str, Any]) -> Dict[str, Any]:
    required = ["bias_present", "bias_instances"]
    for k in required:
        if k not in obj:
            raise ValueError(f"Missing key: {k}")
    if not isinstance(obj["bias_present"], bool):
        raise ValueError("bias_present must be boolean")
    if not isinstance(obj["bias_instances"], list):
        raise ValueError("bias_instances must be list")

    for inst in obj["bias_instances"]:
        if not isinstance(inst, dict):
            raise ValueError("Each bias_instance must be an object")
        for kk in ["span", "bias_type", "notes"]:
            if kk not in inst:
                raise ValueError(f"Missing key in bias_instances: {kk}")

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
            max_tokens=850,
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
    print("BIAS / HEURISTICS CODING RUN")
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

    TEXT_COL = "body"
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

    df_sample[TEXT_COL] = df_sample[TEXT_COL].apply(normalize_text)

    start = time.time()
    results = process_batch(df_sample[TEXT_COL], client, max_workers=2)

    # -----------------------------
    # New columns
    # -----------------------------
    df_sample["bias_present"] = [r["bias_present"] for r in results]
    df_sample["bias_n"] = [safe_count(r.get("bias_instances")) for r in results]

    # Per-bias counts (your list + OTHER)
    BIAS_TYPES = [
        "LAW_OF_SMALL_NUMBERS",
        "AVAILABILITY_HEURISTIC",
        "REPRESENTATIVENESS_HEURISTIC",
        "BASE_RATE_NEGLECT",
        "ATTRIBUTE_SUBSTITUTION",
        "AFFECT_HEURISTIC",
        "CONFIRMATION_BIAS",
        "ILLUSION_OF_VALIDITY",
        "OTHER",
    ]
    for bt in BIAS_TYPES:
        col = f"bias_{bt.lower()}_n"
        df_sample[col] = [
            sum(1 for inst in (r.get("bias_instances") or []) if str(inst.get("bias_type", "")).upper() == bt)
            for r in results
        ]

    # Store full instances
    df_sample["bias_instances_json"] = [
        json.dumps(r.get("bias_instances", []), ensure_ascii=False) for r in results
    ]

    # QA-friendly spans
    df_sample["bias_spans"] = [join_spans(r.get("bias_instances", []), key="span") for r in results]

    # Metadata/provenance
    df_sample["bias_model"] = "llama-3.3-70b-versatile"
    df_sample["bias_prompt_version"] = "v1_biases_kahneman_strict"
    df_sample["bias_max_chars"] = 6000

    output_file = r"data/comments_biases.tsv"
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
