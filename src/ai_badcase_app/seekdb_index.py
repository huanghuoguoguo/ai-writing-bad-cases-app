from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from .models import BadCase, RetrievalHit


DEFAULT_SEEKDB_PATH = Path(".seekdb")
DEFAULT_DATABASE = "ai_badcase"
DEFAULT_COLLECTION = "bad_cases"


class SeekDBUnavailableError(RuntimeError):
    pass


class SeekDBRuntimeError(RuntimeError):
    pass


def _import_seekdb():
    try:
        import pyseekdb  # type: ignore
    except ImportError as exc:
        raise SeekDBUnavailableError(
            "pyseekdb is not installed. Run `uv add pyseekdb` or `uv sync` first."
        ) from exc
    return pyseekdb


def _case_document(case: BadCase) -> str:
    return "\n".join(
        [
            case.label,
            case.description,
            case.why_it_sounds_ai or "",
            " ".join(case.examples),
            case.rewrite_hint,
        ]
    ).strip()


def _build_metadata(case: BadCase) -> dict[str, Any]:
    metadata = asdict(case)
    metadata["matchers"] = [asdict(matcher) for matcher in case.matchers]
    return metadata


def _ensure_database(path: Path, database: str) -> None:
    pyseekdb = _import_seekdb()
    admin = pyseekdb.AdminClient(path=str(path))
    try:
        databases = admin.list_databases()
        if database not in databases:
            admin.create_database(database)
    except Exception as exc:
        raise SeekDBRuntimeError(
            "pyseekdb embedded runtime failed while preparing the database. "
            "The package is installed, but the local embedded engine could not initialize cleanly."
        ) from exc


def get_collection(
    db_path: Path,
    database: str = DEFAULT_DATABASE,
    collection_name: str = DEFAULT_COLLECTION,
):
    pyseekdb = _import_seekdb()
    path = db_path.resolve()
    _ensure_database(path, database)
    client = pyseekdb.Client(path=str(path), database=database)
    try:
        embedding_function = pyseekdb.DefaultEmbeddingFunction()
        return client.get_or_create_collection(
            collection_name,
            embedding_function=embedding_function,
        )
    except Exception as exc:
        raise SeekDBRuntimeError(
            "pyseekdb embedded runtime could not create or open the collection."
        ) from exc


def index_cases(
    cases: list[BadCase],
    db_path: Path = DEFAULT_SEEKDB_PATH,
    database: str = DEFAULT_DATABASE,
    collection_name: str = DEFAULT_COLLECTION,
) -> None:
    collection = get_collection(
        db_path=db_path,
        database=database,
        collection_name=collection_name,
    )
    ids = [case.id for case in cases]
    documents = [_case_document(case) for case in cases]
    metadatas = [_build_metadata(case) for case in cases]
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)


def _where_filter(lang: str | None = None, genres: list[str] | None = None) -> dict[str, Any] | None:
    clauses: list[dict[str, Any]] = []
    if lang:
        clauses.append({"lang": {"$eq": lang}})
    if genres:
        clauses.append({"genres": {"$contains": genres[0]}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _extract_items(raw: dict[str, Any], query_mode: str) -> list[RetrievalHit]:
    ids = raw.get("ids") or [[]]
    documents = raw.get("documents") or [[]]
    metadatas = raw.get("metadatas") or [[]]
    distances = raw.get("distances") or [[]]

    first_ids = ids[0] if ids else []
    first_documents = documents[0] if documents else []
    first_metadatas = metadatas[0] if metadatas else []
    first_distances = distances[0] if distances else []

    hits: list[RetrievalHit] = []
    for index, case_id in enumerate(first_ids):
        metadata = first_metadatas[index] if index < len(first_metadatas) else {}
        distance = first_distances[index] if index < len(first_distances) else 0.0
        score = round(max(0.0, 1.0 - float(distance)), 4)
        hits.append(
            RetrievalHit(
                case_id=case_id,
                label=metadata.get("label", case_id),
                query_mode=query_mode,
                score=score,
                document=first_documents[index] if index < len(first_documents) else None,
                diagnostic_dimensions=metadata.get("diagnostic_dimensions", []),
                rewrite_hint=metadata.get("rewrite_hint", ""),
            )
        )
    return hits


def query_similar(
    text: str,
    db_path: Path = DEFAULT_SEEKDB_PATH,
    database: str = DEFAULT_DATABASE,
    collection_name: str = DEFAULT_COLLECTION,
    top_k: int = 5,
    lang: str | None = None,
    genres: list[str] | None = None,
) -> list[RetrievalHit]:
    collection = get_collection(
        db_path=db_path,
        database=database,
        collection_name=collection_name,
    )
    raw = collection.query(
        query_texts=text,
        n_results=top_k,
        where=_where_filter(lang=lang, genres=genres),
        include=["documents", "metadatas", "distances"],
    )
    return _extract_items(raw, query_mode="vector")


def hybrid_search(
    text: str,
    db_path: Path = DEFAULT_SEEKDB_PATH,
    database: str = DEFAULT_DATABASE,
    collection_name: str = DEFAULT_COLLECTION,
    top_k: int = 5,
    lang: str | None = None,
    genres: list[str] | None = None,
) -> list[RetrievalHit]:
    collection = get_collection(
        db_path=db_path,
        database=database,
        collection_name=collection_name,
    )
    where = _where_filter(lang=lang, genres=genres)
    raw = collection.hybrid_search(
        query={
            "where_document": {"$contains": text[:64]},
            "where": where,
            "n_results": top_k,
        },
        knn={
            "query_texts": text,
            "where": where,
            "n_results": top_k,
        },
        rank={"rrf": {}},
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return _extract_items(raw, query_mode="hybrid")
