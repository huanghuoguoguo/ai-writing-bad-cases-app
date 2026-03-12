from ai_badcase_app.seekdb_index import _extract_items


def test_extract_items_from_query_result():
    raw = {
        "ids": [["case-1"]],
        "documents": [["document body"]],
        "metadatas": [[{"label": "Label", "diagnostic_dimensions": ["x"], "rewrite_hint": "hint"}]],
        "distances": [[0.2]],
    }

    hits = _extract_items(raw, query_mode="vector")

    assert len(hits) == 1
    assert hits[0].case_id == "case-1"
    assert hits[0].label == "Label"
    assert hits[0].score == 0.8
    assert hits[0].diagnostic_dimensions == ["x"]
