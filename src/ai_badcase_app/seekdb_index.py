from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from .models import BadCase


class SeekDBUnavailableError(RuntimeError):
    pass


def _import_seekdb():
    try:
        import pyseekdb  # type: ignore
    except ImportError as exc:
        raise SeekDBUnavailableError(
            "pyseekdb is not installed. Install the optional dependency first."
        ) from exc
    return pyseekdb


def index_cases(cases: list[BadCase], db_path: Path) -> None:
    pyseekdb = _import_seekdb()
    client = pyseekdb.Client(path=str(db_path))
    collection = client.get_or_create_collection("bad_cases")

    docs = []
    metadatas = []
    ids = []
    for case in cases:
        docs.append(
            "\n".join(
                [
                    case.label,
                    case.description,
                    case.why_it_sounds_ai or "",
                    " ".join(case.examples),
                    case.rewrite_hint,
                ]
            ).strip()
        )
        metadatas.append(asdict(case))
        ids.append(case.id)

    collection.upsert(ids=ids, documents=docs, metadatas=metadatas)


def query_similar(text: str, db_path: Path, top_k: int = 5) -> dict:
    pyseekdb = _import_seekdb()
    client = pyseekdb.Client(path=str(db_path))
    collection = client.get_collection("bad_cases")
    return collection.query(query_texts=[text], n_results=top_k)
