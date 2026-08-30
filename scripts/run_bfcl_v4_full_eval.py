"""Run the pinned official BFCL V4 suite with the configured DeepSeek model.

This is a thin adapter around the upstream ``bfcl_eval`` package.  It registers
the newer DeepSeek model identifier used by this repository without changing
the official datasets, execution backends, or scorers.

Run this script with the isolated BFCL environment documented in
``docs/bfcl-v4-full-evaluation.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

MODEL_REGISTRY_NAME = "DeepSeek-V4-Flash-FC"
MODEL_API_NAME = "deepseek-v4-flash"
OFFICIAL_SOURCE = Path("evals/vendor/bfcl_official_source")
DEFAULT_RUN_DIR = Path("evals/runs/bfcl-v4-full-flash-20260820")
DEFAULT_RESULT_DIR = DEFAULT_RUN_DIR / "results"
DEFAULT_SCORE_DIR = DEFAULT_RUN_DIR / "scores"
DEFAULT_AUDIT = DEFAULT_RUN_DIR / "audit.json"


def _register_model() -> None:
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
    if deepseek_key:
        # Upstream's DeepSeek handler calls the OpenAI base constructor before
        # replacing that client with its DeepSeek client.  openai>=3 requires a
        # credential at that intermediate construction point.
        os.environ.setdefault("OPENAI_API_KEY", deepseek_key)

    import httpx2
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
    from bfcl_eval.model_handler.api_inference.openai_response import (
        OpenAIResponsesHandler,
    )
    from openai import OpenAI

    class DeepSeekFlashResponsesHandler(OpenAIResponsesHandler):
        """Use DeepSeek Flash's Responses API without inheriting shell proxies."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            self.client = OpenAI(
                base_url="https://api.deepseek.com",
                api_key=os.environ.get("DEEPSEEK_API_KEY"),
                http_client=httpx2.Client(trust_env=False, timeout=120.0),
            )

    MODEL_CONFIG_MAPPING[MODEL_REGISTRY_NAME] = ModelConfig(
        model_name=MODEL_API_NAME,
        display_name="DeepSeek V4 Flash (Responses FC, local evaluation)",
        url="https://api.deepseek.com",
        org="DeepSeek",
        license="Proprietary",
        model_handler=DeepSeekFlashResponsesHandler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=True,
    )

    if not os.environ.get("SERPAPI_API_KEY"):
        from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.web_search import (
            WebSearchAPI,
        )
        from ddgs import DDGS

        def ddgs_search_engine_query(
            self: WebSearchAPI,
            keywords: str,
            max_results: int | None = 10,
            region: str | None = "wt-wt",
        ) -> list[dict[str, str]] | dict[str, str]:
            try:
                rows = DDGS(proxy=os.environ.get("BFCL_WEB_PROXY")).text(
                    keywords,
                    region=region or "wt-wt",
                    max_results=max_results or 10,
                )
            except Exception as exc:  # pragma: no cover - depends on live search service
                return {"error": f"DuckDuckGo search failed: {exc}"}
            results = []
            for row in rows:
                result = {
                    "title": str(row.get("title", "")),
                    "href": str(row.get("href", "")),
                }
                if self.show_snippet:
                    result["body"] = str(row.get("body", ""))
                results.append(result)
            return results

        WebSearchAPI.search_engine_query = ddgs_search_engine_query


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _official_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(OFFICIAL_SOURCE), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def inventory(output: Path | None) -> dict[str, Any]:
    from bfcl_eval.constants.category_mapping import ALL_SCORING_CATEGORIES
    from bfcl_eval.constants.eval_config import PACKAGE_ROOT, POSSIBLE_ANSWER_PATH, PROMPT_PATH
    from bfcl_eval.utils import is_memory_prereq, load_dataset_entry

    categories: dict[str, dict[str, int]] = {}
    scoring_total = 0
    generation_total = 0
    for category in ALL_SCORING_CATEGORIES:
        entries = load_dataset_entry(category)
        prereqs = sum(is_memory_prereq(entry["id"].rsplit("_", 1)[0]) for entry in entries)
        categories[category] = {
            "scoringEntries": len(entries) - prereqs,
            "prerequisiteEntries": prereqs,
            "generationEntries": len(entries),
        }
        scoring_total += len(entries) - prereqs
        generation_total += len(entries)

    source_files = sorted(PROMPT_PATH.glob("BFCL_v4_*.json"))
    answer_files = sorted(POSSIBLE_ANSWER_PATH.glob("BFCL_v4_*.json"))
    payload: dict[str, Any] = {
        "benchmark": "BFCL V4",
        "officialCommit": _official_commit(),
        "packageRoot": str(PACKAGE_ROOT),
        "modelRegistryName": MODEL_REGISTRY_NAME,
        "modelApiName": MODEL_API_NAME,
        "webSearchBackend": (
            "serpapi-duckduckgo" if os.environ.get("SERPAPI_API_KEY") else "ddgs-duckduckgo"
        ),
        "scoringCategories": list(ALL_SCORING_CATEGORIES),
        "categoryCount": len(ALL_SCORING_CATEGORIES),
        "scoringEntryCount": scoring_total,
        "generationEntryCountIncludingPrerequisites": generation_total,
        "categories": categories,
        "datasetSha256": {path.name: _sha256(path) for path in source_files},
        "groundTruthSha256": {path.name: _sha256(path) for path in answer_files},
    }
    if output is not None:
        _write_json(output, payload)
    return payload


def generate(args: argparse.Namespace) -> None:
    from bfcl_eval._llm_response_generation import main as generation_main

    generation_main(
        SimpleNamespace(
            model=[MODEL_REGISTRY_NAME],
            test_category=args.categories,
            temperature=args.temperature,
            include_input_log=args.include_input_log,
            exclude_state_log=args.exclude_state_log,
            num_gpus=1,
            num_threads=args.threads,
            gpu_memory_utilization=0.0,
            backend="vllm",
            skip_server_setup=True,
            local_model_path=None,
            result_dir=str(args.result_dir.resolve()),
            allow_overwrite=args.overwrite,
            run_ids=args.run_ids,
            enable_lora=False,
            max_lora_rank=None,
            lora_modules=None,
        )
    )


def evaluate(args: argparse.Namespace) -> None:
    from bfcl_eval.eval_checker.eval_runner import main as evaluation_main

    evaluation_main(
        [MODEL_REGISTRY_NAME],
        args.categories,
        str(args.result_dir.resolve()),
        str(args.score_dir.resolve()),
        args.partial,
    )


def audit(args: argparse.Namespace) -> dict[str, Any]:
    from bfcl_eval.constants.category_mapping import ALL_SCORING_CATEGORIES
    from bfcl_eval.constants.eval_config import RESULT_FILE_PATTERN
    from bfcl_eval.utils import (
        extract_test_category,
        is_memory_prereq,
        load_dataset_entry,
    )

    model_root = args.result_dir.resolve() / MODEL_REGISTRY_NAME
    result_files = sorted(model_root.rglob(RESULT_FILE_PATTERN)) if model_root.exists() else []
    actual_by_category: dict[str, list[dict[str, Any]]] = {}
    duplicate_ids: list[str] = []
    error_ids: list[str] = []
    for result_file in result_files:
        category = extract_test_category(result_file)
        rows = [json.loads(line) for line in result_file.read_text(encoding="utf-8").splitlines() if line]
        actual_by_category.setdefault(category, []).extend(rows)
        seen: set[str] = set()
        for row in rows:
            row_id = str(row["id"])
            if row_id in seen:
                duplicate_ids.append(row_id)
            seen.add(row_id)
            if "Error during inference:" in json.dumps(row.get("result"), ensure_ascii=False):
                error_ids.append(row_id)

    category_audit: dict[str, dict[str, Any]] = {}
    missing_total = 0
    extra_total = 0
    for category in ALL_SCORING_CATEGORIES:
        expected_entries = load_dataset_entry(category)
        expected_ids = {
            str(entry["id"])
            for entry in expected_entries
            if not is_memory_prereq(str(entry["id"]).rsplit("_", 1)[0])
        }
        actual_ids = {
            str(row["id"])
            for row in actual_by_category.get(category, [])
            if not is_memory_prereq(str(row["id"]).rsplit("_", 1)[0])
        }
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        missing_total += len(missing)
        extra_total += len(extra)
        category_audit[category] = {
            "expected": len(expected_ids),
            "actual": len(actual_ids),
            "missingCount": len(missing),
            "extraCount": len(extra),
            "missingIds": missing,
            "extraIds": extra,
        }

    payload = {
        "benchmark": "BFCL V4",
        "officialCommit": _official_commit(),
        "model": MODEL_REGISTRY_NAME,
        "resultDirectory": str(args.result_dir.resolve()),
        "complete": missing_total == 0 and extra_total == 0 and not duplicate_ids and not error_ids,
        "missingCount": missing_total,
        "extraCount": extra_total,
        "duplicateCount": len(duplicate_ids),
        "inferenceErrorCount": len(error_ids),
        "duplicateIds": sorted(set(duplicate_ids)),
        "inferenceErrorIds": sorted(set(error_ids)),
        "categories": category_audit,
    }
    _write_json(args.output, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory_parser = subparsers.add_parser("inventory")
    inventory_parser.add_argument("--output", type=Path)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--categories", nargs="+", default=["all_scoring"])
    generate_parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    generate_parser.add_argument("--threads", type=int, default=4)
    generate_parser.add_argument("--temperature", type=float, default=0.001)
    generate_parser.add_argument("--include-input-log", action="store_true")
    generate_parser.add_argument("--exclude-state-log", action="store_true")
    generate_parser.add_argument("--overwrite", action="store_true")
    generate_parser.add_argument("--run-ids", action="store_true")

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--categories", nargs="+", default=["all_scoring"])
    evaluate_parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    evaluate_parser.add_argument("--score-dir", type=Path, default=DEFAULT_SCORE_DIR)
    evaluate_parser.add_argument("--partial", action="store_true")

    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    audit_parser.add_argument("--output", type=Path, default=DEFAULT_AUDIT)
    return parser


def main() -> None:
    args = _parser().parse_args()
    _register_model()
    if args.command == "inventory":
        print(json.dumps(inventory(args.output), ensure_ascii=False, indent=2))
    elif args.command == "generate":
        generate(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "audit":
        print(json.dumps(audit(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
