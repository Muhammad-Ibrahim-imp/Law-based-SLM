# tests/test_dedup.py

from data.dedup import deduplicate_documents, hash_text


def test_identical_documents_are_deduplicated():
    docs = [
        "This is a sample legal document.",
        "This is a sample legal document.",
        "This is a different document.",
    ]
    result = deduplicate_documents(docs)
    assert len(result) == 2
    assert "This is a sample legal document." in result
    assert "This is a different document." in result


def test_no_duplicates_returns_all_documents():
    docs = ["Doc one.", "Doc two.", "Doc three."]
    result = deduplicate_documents(docs)
    assert len(result) == 3


def test_empty_list_returns_empty_list():
    assert deduplicate_documents([]) == []


def test_hash_is_consistent_for_same_text():
    text = "Some legal text."
    assert hash_text(text) == hash_text(text)


def test_hash_differs_for_different_text():
    assert hash_text("Text A") != hash_text("Text B")


def test_first_occurrence_is_kept_in_order():
    docs = ["First.", "Second.", "First.", "Third."]
    result = deduplicate_documents(docs)
    assert result == ["First.", "Second.", "Third."]