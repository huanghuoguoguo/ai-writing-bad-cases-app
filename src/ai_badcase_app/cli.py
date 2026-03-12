from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .library import DEFAULT_LIBRARY_ROOT, load_cases
from .matcher import detect_paragraphs


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
        choices=["text", "json"],
        default="text",
        help="Output format",
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
            lines.append(f"    rewrite_hint: {hit.rewrite_hint}")
        lines.append("")
    return "\n".join(lines).strip()


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    text = input_path.read_text(encoding="utf-8")
    cases = load_cases(
        library_root=Path(args.library_root),
        lang=args.lang,
        genres=args.genres,
    )
    results = detect_paragraphs(text, cases)

    if args.format == "json":
        print(json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2))
        return

    print(_render_text(results))


if __name__ == "__main__":
    main()
