#!/usr/bin/env python3
"""Fetch Wikipedia article snapshots and build unreviewed datasets offline."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from greekttsbench.wikipedia import (
    BOUNDARY, WikipediaClient, WikipediaError, build_dataset, collect_snapshot,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="Save article HTML and revision provenance")
    fetch.add_argument("--language", required=True, help="Wikipedia edition code, e.g. el or fr")
    fetch.add_argument("--title", action="append", default=[], help="Article title; repeatable")
    fetch.add_argument("--titles-file", type=Path, help="UTF-8 file with one title per line")
    fetch.add_argument("--category", action="append", default=[], help="Full category title; direct members only")
    fetch.add_argument("--max-pages", type=int, default=20)
    fetch.add_argument("--user-agent", required=True, help="Descriptive client name and your contact URL/email")
    fetch.add_argument("--snapshot", type=Path, required=True)
    build = commands.add_parser("build", help="Build candidate dataset from a snapshot, offline")
    build.add_argument("--snapshot", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--dataset-id", required=True)
    build.add_argument("--scenario", default="unreviewed", help="Provisional user label; no automatic classification")
    build.add_argument("--split-mode", choices=["punctuation", "paragraph"], default="punctuation")
    build.add_argument("--boundary-pattern", default=BOUNDARY, help="Python split regex; no capturing groups")
    build.add_argument("--min-chars", type=int, default=25)
    build.add_argument("--max-chars", type=int, default=300)
    build.add_argument("--min-words", type=int, default=0, help="Optional rough-token minimum")
    build.add_argument("--max-words", type=int, default=0, help="Optional rough-token maximum; 0 disables")
    build.add_argument("--max-sentences", type=int, default=200)
    build.add_argument("--seed", type=int, default=20260610)
    build.add_argument("--include-pattern", help="Keep candidates matching this Python regex")
    build.add_argument("--exclude-pattern", help="Reject candidates matching this Python regex")
    return parser.parse_args()


def save_new(path: Path, data: dict):
    """Never silently replace a snapshot or an existing dataset."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(content)


def main() -> int:
    args = parse_args()
    try:
        target = args.snapshot if args.command == "fetch" else args.output
        if target.exists():
            raise ValueError(f"Output exists; choose a new path: {target}")
        if args.command == "fetch":
            titles = list(args.title)
            if args.titles_file:
                titles.extend(line.strip() for line in args.titles_file.read_text(encoding="utf-8").splitlines()
                              if line.strip())
            client = WikipediaClient(args.language, args.user_agent)
            data = collect_snapshot(client, titles, args.category, args.max_pages)
            if not data["pages"]:
                raise ValueError(f"No usable articles; no snapshot written. Skips: {data['skipped']}")
            save_new(target, data)
            print(f"Saved {len(data['pages'])} articles, {len(data['skipped'])} skipped: {target}")
        else:
            raw = args.snapshot.read_bytes()
            snapshot = json.loads(raw)
            settings = {key: value for key, value in vars(args).items()
                        if key not in {"command", "snapshot", "output"}}
            data = build_dataset(snapshot, **settings)
            data["collection"]["snapshot_sha256"] = sha256(raw).hexdigest()
            if not data["sentences"]:
                raise ValueError(f"No candidates passed the filters; no dataset written. Counts: {data['collection']['counts']}")
            save_new(target, data)
            print(json.dumps(data["collection"]["counts"], ensure_ascii=False, sort_keys=True))
            print(f"Wrote {target}; candidates require language and provenance review")
        return 0
    except (OSError, ValueError, KeyError, TypeError, re.error, WikipediaError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
