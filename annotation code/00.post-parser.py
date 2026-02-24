# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 22:37:29 2026

@author: cnmkj
"""
import re
import csv
from pathlib import Path

USER_SEP_RE = re.compile(r"^\*{10,}\s*$")
USER_LINE_RE = re.compile(r"^u/([A-Za-z0-9_-]+)\s*$")
SUBREDDIT_RE = re.compile(r"^r/([A-Za-z0-9_]+)(?:\s+icon)?\s*$")
BULLET_TITLE_RE = re.compile(r"^\s*•\s*(.+?)\s*$")
COMMENTED_RE = re.compile(r"^([A-Za-z0-9_-]+)\s+commented\s+(.+?)\s*$")

def clean_body(lines):
    # Strip whitespace and collapse excessive blank lines
    stripped = [l.strip() for l in lines]
    cleaned = []
    prev_blank = False
    for line in stripped:
        if line == "":
            if not prev_blank:
                cleaned.append("")
            prev_blank = True
        else:
            cleaned.append(line)
            prev_blank = False
    return "\n".join(cleaned).strip()

def parse_file(text):
    lines = text.splitlines()
    records = []

    current_author = None
    cur = None

    def flush():
        nonlocal cur
        if cur:
            body = clean_body(cur["body_lines"])
            if body:
                records.append({
                    "author": (cur.get("author") or "").strip(),
                    "subreddit": (cur.get("subreddit") or "").strip(),
                    "thread_title": (cur.get("thread_title") or "").strip(),
                    "commented_when": (cur.get("commented_when") or "").strip(),
                    "body": body
                })
        cur = None

    def start_comment():
        nonlocal cur
        flush()
        cur = {
            "author": current_author,
            "subreddit": None,
            "thread_title": None,
            "commented_when": None,
            "body_lines": []
        }

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line == "":
            if cur:
                cur["body_lines"].append("")
            i += 1
            continue

        if USER_SEP_RE.match(line):
            flush()
            i += 1
            continue

        m_user = USER_LINE_RE.match(line)
        if m_user:
            current_author = m_user.group(1)
            i += 1
            continue

        if line in {"Comments", "Overview", "Posts"}:
            i += 1
            continue

        m_sub = SUBREDDIT_RE.match(line)
        if m_sub:
            start_comment()
            cur["subreddit"] = m_sub.group(1)
            i += 1
            continue

        m_title = BULLET_TITLE_RE.match(line)
        if m_title and cur:
            cur["thread_title"] = m_title.group(1)
            i += 1
            continue

        m_commented = COMMENTED_RE.match(line)
        if m_commented and cur:
            cur["commented_when"] = m_commented.group(2)
            i += 1
            continue

        if cur:
            cur["body_lines"].append(lines[i])

        i += 1

    flush()
    return records

def write_tsv(records, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["author", "subreddit", "thread_title", "commented_when", "body"],
            delimiter="\t",
            quoting=csv.QUOTE_MINIMAL
        )
        writer.writeheader()
        writer.writerows(records)

if __name__ == "__main__":
    text = Path("data/CMV-LLM-Posts.txt").read_text(encoding="utf-8", errors="replace")
    records = parse_file(text)
    write_tsv(records, "data/CMV-LLM-Posts_parsed.tsv")
    print(f"Wrote {len(records)} comments to comments.tsv")
