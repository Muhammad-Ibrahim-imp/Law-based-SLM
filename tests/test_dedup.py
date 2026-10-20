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

# ---------------------------------------------------------------- legal pretraining helpers
import numpy as np  # noqa: E402

from data.dedup import minhash_signature, near_duplicate_clusters, ngram_hashes, normalize_for_hashing  # noqa: E402

BASE = " ".join(f"The court considered argument number {i} and found it persuasive in part." for i in range(80))


def test_normalize_for_hashing():
    assert normalize_for_hashing("  The COURT,  held:\n\n(a) ‘yes’!! ") == "the court held a yes"
    assert normalize_for_hashing("Smith v. Jones") == normalize_for_hashing("smith  v jones")


def test_minhash_similarity_tracks_overlap():
    near = BASE.replace("number 3 ", "number three ", 1)
    other = " ".join(f"Unrelated statute text clause {i} about fishing licences." for i in range(80))
    a, b, c = (minhash_signature(t) for t in (BASE, near, other))
    assert a.dtype == np.uint32 and a.shape == (128,)
    assert (a == b).mean() > 0.9 and (a == c).mean() < 0.1
    assert (minhash_signature(BASE) == a).all()            # deterministic


def test_near_duplicate_clusters_and_exact_keys():
    texts = [BASE, BASE.replace("number 3 ", "number three ", 1), "Different text entirely " * 40, BASE]
    sigs = np.stack([minhash_signature(t) for t in texts])
    keys = np.array([1, 2, 3, 1], np.uint64)
    roots, stats = near_duplicate_clusters(sigs, 0.8, exact_keys=keys)
    assert roots.tolist() == [0, 0, 2, 0] and stats["exact_pairs"] == 1


def test_ngram_hashes():
    g = ngram_hashes(BASE, 13)
    assert len(g) > 0 and (np.diff(g.astype(np.float64)) >= 0).all()
    assert len(ngram_hashes("too short to have thirteen words", 13)) == 0
    assert np.isin(ngram_hashes(" ".join(BASE.split()[:40]), 13), g).all()


def test_longest_runs():
    from data.dedup import longest_runs
    pos = np.array([3, 4, 5, 9, 0, 1, 7], np.int64)      # doc0: run 3; doc1: empty; doc2: run 2
    assert longest_runs(pos, np.array([0, 4, 4, 7])).tolist() == [3, 0, 2]
    assert longest_runs(np.zeros(0, np.int64), np.array([0, 0])).tolist() == [0]
