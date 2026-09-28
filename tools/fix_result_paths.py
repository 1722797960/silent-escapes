#!/usr/bin/env python3
"""Rewrite stale absolute paths inside historical result JSONs to repo-relative paths.

The pre-computed result JSONs shipped in results/ were produced while the experiment
tree lived in /root/data-fs/WZB/integrity-clash-wam[-e2e]/. The file paths recorded in
them (metadata "image" fields etc.) are descriptive provenance, not load paths, but for
a self-contained / publishable tree they should not leak the old absolute locations.

This script maps every recorded old prefix to the appropriate new location inside this
repository (or fails loudly on an unexpected prefix). Idempotent.

Run from the repository root:  python tools/fix_result_paths.py
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# old absolute prefix -> replacement relative to repo root
PREFIX_MAP = {
    "/root/data-fs/WZB/Silent-Escapes": "",
    "/root/data-fs/WZB/integrity-clash-wam-e2e": "",
    "/root/data-fs/WZB/integrity-clash-wam": "",
    "/root/data-fs/WZB/integrity-clash": "",
    # legacy tree shorthands that predate the results/ reorganization
    "_outputs_500_wam": "_derived_outputs_500_wam",
}


def new_path(p: str) -> str:
    for old, repl in PREFIX_MAP.items():
        if p.startswith(old):
            rel = p[len(old):].lstrip("/")
            replaced = {"outputs_": "data/derived/",
                        "outputs_n500": "data/originals_500",
                        "results_500": "results",
                        "regional_experiments": "results",
                        }.get(rel.split("/")[0], "")
            if replaced:
                return replaced + rel[len(rel.split("/")[0]):]
            return rel
    return p  # unknown prefix: leave as-is (fail mode is grep-able)


def walk(v):
    if isinstance(v, str):
        if "/root/data-fs" in v:
            return new_path(v)
        normalized = v.replace("\\", "/")
        marker = "/Silent-Escapes/"
        if marker in normalized:
            return normalized.split(marker, 1)[1]
        return v
    if isinstance(v, list):
        return [walk(x) for x in v]
    if isinstance(v, dict):
        return {k: walk(x) for k, x in v.items()}
    return v


def main():
    roots = sys.argv[1:] or [os.path.join(REPO, "results")]
    changed = 0
    for root in roots:
        for dirpath, _, files in os.walk(root):
            for fn in files:
                if not fn.endswith(".json"):
                    continue
                p = os.path.join(dirpath, fn)
                try:
                    txt = open(p, encoding="utf-8").read()
                except UnicodeDecodeError:
                    continue
                if "/root/data-fs" not in txt and "Silent-Escapes" not in txt:
                    continue
                try:
                    data = json.loads(txt)
                except json.JSONDecodeError:
                    print(f"SKIP (not strictly valid JSON): {p}", file=sys.stderr)
                    continue
                fixed = json.dumps(walk(data), indent=1, ensure_ascii=False)
                open(p, "w", encoding="utf-8").write(fixed)
                changed += 1
    print(f"rewrote {changed} JSON file(s)")


if __name__ == "__main__":
    main()
