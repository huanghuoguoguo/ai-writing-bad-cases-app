from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .analyzer import analyze_text
from .library import DEFAULT_LIBRARY_ROOT, load_cases
from .matcher import MatcherConfig, detect_paragraphs, split_paragraphs, split_sentences
from .models import ParagraphResult
from .seekdb_index import (
    DEFAULT_COLLECTION,
    DEFAULT_DATABASE,
    DEFAULT_SEEKDB_PATH,
    SeekDBRuntimeError,
    SeekDBUnavailableError,
    hybrid_search,
    index_cases,
    query_similar,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect AI writing bad cases in text")
    parser.add_argument("--input", required=True, help="Path to a text or markdown file")
    parser.add_argument(
        "--library-root",
        default=str(DEFAULT_LIBRARY_ROOT),
        help="Path to the JSON bad case library root",
    )
    parser.add_argument("--lang", default="zh", help="Language code")
    parser.add_argument(
        "--genre",
        action="append",
        dest="genres",
        help="Genre filter, can be provided more than once",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json", "legacy-json"],
        default="text",
        help="Output format",
    )
    parser.add_argument(
        "--seekdb",
        action="store_true",
        help="Enable pyseekdb retrieval in addition to rule matching",
    )
    parser.add_argument(
        "--rebuild-seekdb-index",
        action="store_true",
        help="Build or refresh the local embedded SeekDB index before querying",
    )
    parser.add_argument(
        "--seekdb-path",
        default=str(DEFAULT_SEEKDB_PATH),
        help="Path to the embedded SeekDB working directory",
    )
    parser.add_argument(
        "--seekdb-database",
        default=DEFAULT_DATABASE,
        help="Embedded SeekDB database name",
    )
    parser.add_argument(
        "--seekdb-collection",
        default=DEFAULT_COLLECTION,
        help="SeekDB collection name",
    )
    parser.add_argument(
        "--seekdb-top-k",
        type=int,
        default=5,
        help="Number of retrieval hits to keep per paragraph",
    )
    parser.add_argument(
        "--seekdb-mode",
        choices=["vector", "hybrid"],
        default="hybrid",
        help="SeekDB retrieval mode (default: hybrid)",
    )
    parser.add_argument(
        "--fuzzy-threshold",
        type=int,
        default=75,
        help="Fuzzy matching threshold (0-100, default: 75)",
    )
    parser.add_argument(
        "--fuzzy-algorithm",
        choices=["ratio", "partial_ratio", "token_sort_ratio", "token_set_ratio"],
        default="ratio",
        help="Fuzzy matching algorithm (default: ratio)",
    )
    parser.add_argument(
        "--enable-perplexity",
        action="store_true",
        help="Enable optional perplexity-based probability signals",
    )
    return parser


def _render_text(results) -> str:
    lines: list[str] = []
    for result in results:
        lines.append(f"[paragraph {result.paragraph_index}] score={result.score}")
        lines.append(result.text)
        for hit in result.hits:
            lines.append(
                f"  - {hit.label} ({hit.matcher_type}, confidence={hit.confidence}) => {hit.matched_text}"
            )
            if hit.diagnostic_dimensions:
                lines.append(f"    dimensions: {', '.join(hit.diagnostic_dimensions)}")
            lines.append(f"    rewrite_hint: {hit.rewrite_hint}")
        for hit in result.retrieval_hits:
            lines.append(
                f"  - [seekdb/{hit.query_mode}] {hit.label} (score={hit.score})"
            )
            lines.append(f"    rewrite_hint: {hit.rewrite_hint}")
        lines.append("")
    return "\n".join(lines).strip()


def _merge_results(
    text: str,
    rule_results: list[ParagraphResult],
    retrieval_map: dict[int, list],
) -> list[ParagraphResult]:
    rule_map = {result.paragraph_index: result for result in rule_results}
    merged: list[ParagraphResult] = []

    for index, paragraph in enumerate(split_paragraphs(text)):
        rule_result = rule_map.get(index)
        retrieval_hits = retrieval_map.get(index, [])
        if not rule_result and not retrieval_hits:
            continue

        if rule_result:
            base_score = rule_result.score
            hits = rule_result.hits
        else:
            base_score = 0.0
            hits = []

        retrieval_score = retrieval_hits[0].score if retrieval_hits else 0.0
        merged.append(
            ParagraphResult(
                paragraph_index=index,
                text=paragraph,
                score=round(max(base_score, retrieval_score), 4),
                hits=hits,
                retrieval_hits=retrieval_hits,
            )
        )

    return sorted(merged, key=lambda item: item.score, reverse=True)


def _query_retrieval(args, chunk: str):
    if args.seekdb_mode == "hybrid":
        return hybrid_search(
            text=chunk,
            db_path=Path(args.seekdb_path),
            database=args.seekdb_database,
            collection_name=args.seekdb_collection,
            top_k=args.seekdb_top_k,
            lang=args.lang,
            genres=args.genres,
        )
    return query_similar(
        text=chunk,
        db_path=Path(args.seekdb_path),
        database=args.seekdb_database,
        collection_name=args.seekdb_collection,
        top_k=args.seekdb_top_k,
        lang=args.lang,
        genres=args.genres,
    )


def _aggregate_retrieval_hits(args, paragraph: str):
    merged: dict[str, object] = {}
    for sentence in split_sentences(paragraph):
        retrieval_hits = _query_retrieval(args, sentence)
        for hit in retrieval_hits:
            existing = merged.get(hit.case_id)
            if existing is None or hit.score > existing.score:
                merged[hit.case_id] = hit

    return sorted(merged.values(), key=lambda item: item.score, reverse=True)[: args.seekdb_top_k]


def _run_legacy_detection(args, text: str) -> list[ParagraphResult]:
    cases = load_cases(
        library_root=Path(args.library_root),
        lang=args.lang,
        genres=args.genres,
    )

    matcher_config = MatcherConfig(
        fuzzy_threshold=args.fuzzy_threshold,
        fuzzy_algorithm=args.fuzzy_algorithm,
    )
    results = detect_paragraphs(text, cases, config=matcher_config)

    if args.rebuild_seekdb_index or args.seekdb:
        try:
            index_cases(
                cases=cases,
                db_path=Path(args.seekdb_path),
                database=args.seekdb_database,
                collection_name=args.seekdb_collection,
            )
        except (SeekDBUnavailableError, SeekDBRuntimeError) as exc:
            raise SystemExit(f"SeekDB index build failed: {exc}") from exc

    if args.seekdb:
        retrieval_map = {}
        paragraphs = split_paragraphs(text)
        for index, paragraph in enumerate(paragraphs):
            try:
                retrieval_hits = _aggregate_retrieval_hits(args, paragraph)
            except (SeekDBUnavailableError, SeekDBRuntimeError) as exc:
                raise SystemExit(f"SeekDB query failed: {exc}") from exc

            if retrieval_hits:
                retrieval_map[index] = retrieval_hits

        results = _merge_results(text, results, retrieval_map)

    return results


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    text = input_path.read_text(encoding="utf-8")

    if args.format == "json":
        report = analyze_text(
            text,
            library_root=Path(args.library_root),
            lang=args.lang,
            genres=args.genres,
            enable_perplexity=args.enable_perplexity,
        )
        print(report.to_json())
        return

    results = _run_legacy_detection(args, text)

    if args.format == "legacy-json":
        print(json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2))
        return

    print(_render_text(results))


if __name__ == "__main__":
    main()
