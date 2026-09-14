#!/usr/bin/env python3
"""Portable, dependency-free GitHub retrieval demo.

This demo intentionally does not import nanobot.  It validates the candidate
retrieval policy before the policy is moved into the /rere workflow:

    query variants -> GitHub Search -> global dedup -> README evidence gate
    -> implementation-only shortlist

It uses only Python's standard library.  Set GITHUB_TOKEN for a normal GitHub
API rate limit, or pass --token explicitly.  The output JSON is an audit
artifact: every candidate has query hits, README evidence, and a rejection
reason when it is not implementation-quality.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

GITHUB_API = "https://api.github.com"
DEFAULT_DIRECTION = "下载并分析五个超声图像重建相关的项目"
DEFAULT_QUERIES = [
    "ultrasound image reconstruction implementation",
    "ultrasound beamforming reconstruction code",
    "ultrasound tomography reconstruction python",
    "ultrasonic inverse imaging reconstruction",
]

MODALITY_TERMS = (
    "ultrasound",
    "ultrasonic",
    "ultrasonography",
    "sonography",
    "超声",
)
TASK_TERMS = (
    "image reconstruction",
    "reconstruction",
    "beamforming",
    "tomography",
    "inverse imaging",
    "reconstruct",
    "重建",
)
IMPLEMENTATION_TERMS = (
    "install",
    "pip install",
    "conda install",
    "docker",
    "quick start",
    "usage",
    "run",
    "inference",
    "training",
    "train",
    "python ",
    "pytorch",
    "tensorflow",
    "jax",
    "cli",
    "command line",
    "requirements.txt",
    "setup.py",
    "pyproject.toml",
    "src/",
    "scripts/",
)
DIRECT_IMPLEMENTATION_TERMS = (
    "pip install",
    "conda install",
    "uv pip",
    "git clone",
    "requirements.txt",
    "setup.py",
    "pyproject.toml",
    "python ",
    "python3",
    "matlab",
    "cuda",
    "jupyter",
    "quickstart",
    "quick start",
    "implementation",
    "implementations",
    "library",
    "toolbox",
    "framework",
    "source code",
    "sample data",
    "example code",
    "implemented in",
    ".py",
    ".m",
    "/python/",
    "/matlab/",
)
RESOURCE_TERMS = (
    "awesome",
    "papers",
    "paper list",
    "bibliography",
    "survey",
    "resource collection",
    "curated collection",
    "collection list",
    "list of tools",
    "collection of book",
    "source code collection",
    "resources",
    "daily paper",
    "literature review",
)


@dataclass
class Candidate:
    full_name: str
    url: str
    description: str
    stars: int
    language: str | None
    topics: list[str]
    license: str | None
    query_hits: list[str] = field(default_factory=list)
    readme_excerpt: str = ""
    modality_terms: list[str] = field(default_factory=list)
    task_terms: list[str] = field(default_factory=list)
    implementation_terms: list[str] = field(default_factory=list)
    direct_implementation_terms: list[str] = field(default_factory=list)
    resource_terms: list[str] = field(default_factory=list)
    accepted: bool = False
    rejection_reason: str = ""
    evidence_score: int = 0


def github_json(path: str, token: str, params: dict[str, str] | None = None) -> dict[str, Any]:
    url = f"{GITHUB_API}{path}"
    if params:
        url = f"{url}?{urlencode(params)}"
    request = Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "github-research-agent-portable-demo",
            **({"Authorization": f"Bearer {token}"} if token else {}),
        },
    )
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"GitHub request failed for {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"GitHub returned a non-object response for {path}")
    return payload


def candidate_from_item(item: dict[str, Any], query: str) -> Candidate | None:
    full_name = item.get("full_name")
    html_url = item.get("html_url")
    if not isinstance(full_name, str) or not isinstance(html_url, str):
        return None
    license_value = item.get("license")
    license_name = None
    if isinstance(license_value, dict):
        license_name = str(license_value.get("spdx_id") or license_value.get("name") or "") or None
    topics_value = item.get("topics")
    topics = [str(value) for value in topics_value] if isinstance(topics_value, list) else []
    return Candidate(
        full_name=full_name,
        url=html_url,
        description=str(item.get("description") or ""),
        stars=int(item.get("stargazers_count") or 0),
        language=str(item["language"]) if item.get("language") else None,
        topics=topics,
        license=license_name,
        query_hits=[query],
    )


def search_query(query: str, per_query: int, token: str) -> list[Candidate]:
    payload = github_json(
        "/search/repositories",
        token,
        {
            "q": f"{query} in:name,description,readme archived:false",
            "sort": "stars",
            "order": "desc",
            "per_page": str(per_query),
        },
    )
    items = payload.get("items")
    if not isinstance(items, list):
        return []
    return [
        candidate
        for item in items
        if isinstance(item, dict)
        for candidate in [candidate_from_item(item, query)]
        if candidate is not None
    ]


def fetch_readme(candidate: Candidate, token: str, max_chars: int) -> str:
    payload = github_json(f"/repos/{candidate.full_name}/readme", token)
    content = payload.get("content")
    if not isinstance(content, str):
        return ""
    try:
        text = base64.b64decode(content.encode("ascii"), validate=False).decode(
            "utf-8", errors="replace"
        )
    except (ValueError, UnicodeError):
        return ""
    return re.sub(r"\r\n?", "\n", text).strip()[:max_chars]


def terms_in(text: str, terms: tuple[str, ...]) -> list[str]:
    lowered = text.casefold()
    return [term for term in terms if term.casefold() in lowered]


def classify(candidate: Candidate) -> None:
    metadata = " ".join(
        [candidate.full_name, candidate.description, candidate.language or "", *candidate.topics]
    )
    evidence = " ".join([metadata, candidate.readme_excerpt])
    candidate.modality_terms = terms_in(evidence, MODALITY_TERMS)
    candidate.task_terms = terms_in(evidence, TASK_TERMS)
    candidate.implementation_terms = terms_in(evidence, IMPLEMENTATION_TERMS)
    candidate.direct_implementation_terms = terms_in(evidence, DIRECT_IMPLEMENTATION_TERMS)
    candidate.resource_terms = terms_in(evidence, RESOURCE_TERMS)

    first_section = re.sub(r"\s+", " ", evidence[:1_200]).casefold()
    list_title = any(
        marker in first_section
        for marker in (
            "awesome",
            "curated list",
            "curated collection",
            "collection list",
            "list of tools",
            "collection of book",
            "source code collection",
            "paper list",
            "bibliography",
        )
    )
    dataset_only = "dataset" in first_section and len(candidate.direct_implementation_terms) < 3
    # A resource/list or dataset repository is not a code implementation merely
    # because its prose mentions training, inference, Python, or code links.
    resource_only = list_title or dataset_only
    # Dataset/tooling projects remain valid candidates, but a direct
    # reconstruction implementation should outrank a data-release repository
    # when the requested count is limited.
    dataset_penalty = 5 if "dataset" in first_section else 0
    candidate.evidence_score = (
        (3 if candidate.modality_terms else 0)
        + (3 if candidate.task_terms else 0)
        + min(4, len(candidate.direct_implementation_terms))
        - dataset_penalty
        - (5 if resource_only else 0)
    )
    if not candidate.modality_terms:
        candidate.rejection_reason = "README/metadata lacks an ultrasound modality signal"
    elif not candidate.task_terms:
        candidate.rejection_reason = "README/metadata lacks an image-reconstruction task signal"
    elif resource_only:
        candidate.rejection_reason = (
            "resource/paper/list repository without enough concrete install or run evidence"
        )
    elif len(candidate.direct_implementation_terms) < 2:
        candidate.rejection_reason = "insufficient evidence of source-code execution"
    else:
        candidate.accepted = True


def deduplicate(candidates: list[Candidate]) -> list[Candidate]:
    unique: dict[str, Candidate] = {}
    for candidate in candidates:
        key = candidate.full_name.casefold()
        current = unique.get(key)
        if current is None:
            unique[key] = candidate
            continue
        current.query_hits = sorted(set(current.query_hits + candidate.query_hits))
        if candidate.stars > current.stars:
            candidate.query_hits = current.query_hits
            unique[key] = candidate
    return list(unique.values())


def build_report(
    direction: str,
    queries: list[str],
    candidates: list[Candidate],
    selected: list[Candidate],
    output: Path,
) -> None:
    record = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "direction": direction,
        "queries": queries,
        "recalled_count": len(candidates),
        "accepted_count": sum(candidate.accepted for candidate in candidates),
        "selected_count": len(selected),
        "selected": [asdict(candidate) for candidate in selected],
        "candidates": [asdict(candidate) for candidate in candidates],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: record[key] for key in record if key not in {"candidates", "selected"}}, ensure_ascii=False, indent=2))
    print("\nSELECTED IMPLEMENTATIONS")
    for index, candidate in enumerate(selected, 1):
        print(
            f"{index}. {candidate.full_name} | score={candidate.evidence_score} | "
            f"queries={len(candidate.query_hits)} | {candidate.url}"
        )
        print(f"   modality={candidate.modality_terms}; task={candidate.task_terms}")
        print(f"   implementation={candidate.implementation_terms[:8]}")
    print("\nREJECTED OR BORDERLINE CANDIDATES")
    for candidate in sorted(candidates, key=lambda item: item.evidence_score, reverse=True):
        if not candidate.accepted:
            print(f"- {candidate.full_name}: {candidate.rejection_reason}")
    print(f"\nAudit record: {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", default=DEFAULT_DIRECTION)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--per-query", type=int, default=25)
    parser.add_argument("--max-readme-chars", type=int, default=8_000)
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN", ""))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".nanobot/repository-agent/demo/ultrasound-retrieval.json"),
    )
    args = parser.parse_args()
    if args.count < 1 or args.per_query < 1:
        parser.error("--count and --per-query must be positive")

    queries = DEFAULT_QUERIES if "超声" in args.direction or "ultrasound" in args.direction.casefold() else [
        f"{args.direction} implementation",
        f"{args.direction} source code",
    ]
    print(f"Recalling up to {len(queries) * args.per_query} candidates from GitHub…", file=sys.stderr)
    recalled: list[Candidate] = []
    with ThreadPoolExecutor(max_workers=len(queries)) as executor:
        futures = [executor.submit(search_query, query, args.per_query, args.token) for query in queries]
        for future in as_completed(futures):
            try:
                recalled.extend(future.result())
            except RuntimeError as exc:
                print(f"WARNING: {exc}", file=sys.stderr)
    candidates = deduplicate(recalled)
    print(f"Deduplicated to {len(candidates)} candidates; fetching README evidence…", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(fetch_readme, candidate, args.token, args.max_readme_chars): candidate
            for candidate in candidates
        }
        for future in as_completed(futures):
            candidate = futures[future]
            try:
                candidate.readme_excerpt = future.result()
            except RuntimeError as exc:
                candidate.readme_excerpt = ""
                candidate.rejection_reason = f"README fetch failed: {exc}"
            classify(candidate)

    accepted = sorted(
        [candidate for candidate in candidates if candidate.accepted],
        key=lambda candidate: (
            -candidate.evidence_score,
            -len(candidate.query_hits),
            candidate.full_name.casefold(),
        ),
    )
    selected = accepted[: args.count]
    build_report(args.direction, queries, candidates, selected, args.output)
    if len(selected) < args.count:
        print(
            f"WARNING: only {len(selected)} implementation-quality repositories were found; "
            f"requested {args.count}. No resource lists were used to fill the gap.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
