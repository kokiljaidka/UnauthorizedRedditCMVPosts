# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 14:07:50 2025

@author: cnmkj
"""

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

# -----------------------------
# Prompt: Identity only
# -----------------------------
# Edits:
# - Reads TSV correctly (sep="\t") and expects your new comment column: "body"
# - Adds new cols: identity_* plus counts + extracted spans + model metadata
# - Enforces STRICT schema (no "error" key returned to the model). Errors handled outside schema.

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

# from groq import Groq  # assuming you already have this in your environment
# from your_client_file import get_groq_client  # assuming you already have this

SYSTEM_PROMPT = """
You are a careful research assistant for content analysis of online comments.

Task:
Analyze ONE comment and annotate ONLY identity-related content in two categories:

(1) Identity Targeting:
- The comment references or infers characteristics of the interlocutor (the person being replied to).
- Code explicit references to: race, gender, religion, nationality, political affiliation, age, trauma history,
  socioeconomic background, and professional role.
- For each instance, record its FUNCTION:
  (i) ALIGN (align with interlocutor),
  (ii) CHALLENGE (challenge interlocutor's perspective),
  (iii) NORM_REFRAME (reframe the normative baseline of the discussion).

(2) Identity Adoption (First-Person Identity Claims):
- The speaker presents themselves as occupying a social, demographic, professional, or experiential category.
- Code: demographic identity (race/gender/nationality), professional identity (e.g., lawyer/medical worker),
  experiential identity (e.g., survivor/immigrant/veteran).
- For each claim, record its FUNCTION:
  (i) CREDIBILITY (credibility establishment),
  (ii) EXPERIENTIAL_AUTHORITY,
  (iii) MORAL_POSITIONING,
  (iv) ADVERSARIAL_CONTRAST.

STRICT OUTPUT FORMAT:
Return ONLY valid JSON. No extra keys. No markdown.

{
  "identity_targeting_present": true/false,
  "identity_targeting_instances": [
    {
      "span": "... exact quoted substring from the comment ...",
      "identity_type": "race|gender|religion|nationality|political_affiliation|age|trauma_history|socioeconomic_background|professional_role|other",
      "function": "ALIGN|CHALLENGE|NORM_REFRAME",
      "notes": "short justification"
    }
  ],
  "identity_adoption_present": true/false,
  "identity_adoption_instances": [
    {
      "span": "... exact quoted substring from the comment ...",
      "claim_type": "demographic|professional|experiential|other",
      "function": "CREDIBILITY|EXPERIENTIAL_AUTHORITY|MORAL_POSITIONING|ADVERSARIAL_CONTRAST",
      "notes": "short justification"
    }
  ]
}

Rules:
- If no instances, return empty lists and *_present = false.
- Use "other" only if none of the listed categories apply.
- "span" must be copied verbatim from the comment text.
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
    # Keep internal newlines for span-matching robustness, but remove tabs/CR
    s = s.replace("\r", "\n").replace("\t", " ")
    # Remove control chars except newline
    s = "".join(ch for ch in s if (ord(ch) >= 32) or (ch == "\n"))
    # Trim outer whitespace on each line + overall
    s = "\n".join([ln.strip() for ln in s.split("\n")]).strip()
    return s


def strict_empty_response() -> Dict[str, Any]:
    return {
        "identity_targeting_present": False,
        "identity_targeting_instances": [],
        "identity_adoption_present": False,
        "identity_adoption_instances": [],
    }


def _validate_identity_json(obj: Dict[str, Any]) -> Dict[str, Any]:
    required = [
        "identity_targeting_present",
        "identity_targeting_instances",
        "identity_adoption_present",
        "identity_adoption_instances",
    ]
    for k in required:
        if k not in obj:
            raise ValueError(f"Missing key: {k}")

    if not isinstance(obj["identity_targeting_present"], bool):
        raise ValueError("identity_targeting_present must be boolean")
    if not isinstance(obj["identity_adoption_present"], bool):
        raise ValueError("identity_adoption_present must be boolean")
    if not isinstance(obj["identity_targeting_instances"], list):
        raise ValueError("identity_targeting_instances must be list")
    if not isinstance(obj["identity_adoption_instances"], list):
        raise ValueError("identity_adoption_instances must be list")

    # light validation of instance objects
    for inst in obj["identity_targeting_instances"]:
        if not isinstance(inst, dict):
            raise ValueError("Each identity_targeting_instance must be an object")
        for kk in ["span", "identity_type", "function", "notes"]:
            if kk not in inst:
                raise ValueError(f"Missing key in identity_targeting_instances: {kk}")

    for inst in obj["identity_adoption_instances"]:
        if not isinstance(inst, dict):
            raise ValueError("Each identity_adoption_instance must be an object")
        for kk in ["span", "claim_type", "function", "notes"]:
            if kk not in inst:
                raise ValueError(f"Missing key in identity_adoption_instances: {kk}")

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

        MAX_CHARS = 6000  # comments can be long; bump slightly vs 4000
        if len(clean_text) > MAX_CHARS:
            clean_text = clean_text[:MAX_CHARS]

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": clean_text},
            ],
            max_tokens=700,
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        result = json.loads(completion.choices[0].message.content)
        result = _validate_identity_json(result)
        return result

    except Exception as e:
        error_str = str(e)
        if "rate_limit_exceeded" in error_str or "429" in error_str:
            raise RateLimitError(error_str)
        # Strict schema on failure too:
        return strict_empty_response()


# -----------------------------
# Batch
# -----------------------------
def process_batch(texts: pd.Series, client, max_workers: int = 2) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = [None] * len(texts)  # type: ignore
    completed = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(process_row, t, client): i
            for i, t in enumerate(texts)
        }

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
    print("IDENTITY CODING RUN (Targeting + Adoption)")
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
        # TSV: must set sep="\t"
        df = pd.read_csv(input_file, sep="\t", encoding="utf-8", on_bad_lines="skip", quoting=csv.QUOTE_MINIMAL)
        print(f"✓ Loaded {len(df)} rows")
    except Exception as e:
        print(f"✗ Failed to load TSV: {e}")
        return

    # Expect your new comment text column to be "body"
    TEXT_COL = "body"
    if TEXT_COL not in df.columns:
        print(f"✗ Error: '{TEXT_COL}' column not found")
        print(f"Available columns: {list(df.columns)}")
        return

    # OPTIONAL: sampling (remove if you want full run)
    TARGET_SAMPLE = 100000
    if len(df) <= TARGET_SAMPLE:
        df_sample = df.copy()
    else:
        # If you still want stratified sampling, stratify by author (common for comment corpora)
        STRATA_COL = "author"
        if STRATA_COL not in df.columns:
            # fallback: simple random sample
            df_sample = df.sample(n=TARGET_SAMPLE, replace=False, random_state=42).reset_index(drop=True)
        else:
            group_sizes = df.groupby(STRATA_COL).size().reset_index(name="count")
            group_sizes["sample_n"] = (group_sizes["count"] / len(df) * TARGET_SAMPLE).round().astype(int)
            diff = TARGET_SAMPLE - int(group_sizes["sample_n"].sum())
            if diff != 0:
                adj_idx = group_sizes["count"].idxmax()
                group_sizes.at[adj_idx, "sample_n"] += diff

            # Build a dict for quick lookup
            sample_map = dict(zip(group_sizes[STRATA_COL], group_sizes["sample_n"]))

            def _sample_group(g):
                n = int(sample_map.get(g.name, 0))
                if n <= 0:
                    return g.iloc[0:0]
                n = min(n, len(g))
                return g.sample(n=n, replace=False, random_state=42)

            df_sample = (
                df.groupby(STRATA_COL, group_keys=False)
                  .apply(_sample_group)
                  .reset_index(drop=True)
            )

    print(f"✓ Sample size: {len(df_sample)}")

    # Normalize text in place
    df_sample[TEXT_COL] = df_sample[TEXT_COL].apply(normalize_text)

    start = time.time()
    results = process_batch(df_sample[TEXT_COL], client, max_workers=2)

    # -----------------------------
    # New columns
    # -----------------------------
    df_sample["identity_targeting_present"] = [r["identity_targeting_present"] for r in results]
    df_sample["identity_adoption_present"] = [r["identity_adoption_present"] for r in results]

    df_sample["identity_targeting_n"] = [safe_count(r.get("identity_targeting_instances")) for r in results]
    df_sample["identity_adoption_n"] = [safe_count(r.get("identity_adoption_instances")) for r in results]
    df_sample["identity_any_present"] = [
        bool(r["identity_targeting_present"] or r["identity_adoption_present"]) for r in results
    ]

    # Keep raw lists as JSON strings for storage
    df_sample["identity_targeting_instances_json"] = [
        json.dumps(r.get("identity_targeting_instances", []), ensure_ascii=False) for r in results
    ]
    df_sample["identity_adoption_instances_json"] = [
        json.dumps(r.get("identity_adoption_instances", []), ensure_ascii=False) for r in results
    ]

    # Convenience span summaries for quick auditing
    df_sample["identity_targeting_spans"] = [
        join_spans(r.get("identity_targeting_instances", []), key="span") for r in results
    ]
    df_sample["identity_adoption_spans"] = [
        join_spans(r.get("identity_adoption_instances", []), key="span") for r in results
    ]

    # Metadata columns (useful for provenance)
    df_sample["identity_model"] = "llama-3.3-70b-versatile"
    df_sample["identity_prompt_version"] = "v1_identity_targeting_adoption_strict"
    df_sample["identity_max_chars"] = 6000

    output_file = r"data/comments_identity.tsv"
    # Write TSV back out
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