# data/dedup.py

import hashlib


def hash_text(text: str) -> str:
    """Produce a fingerprint for a piece of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def deduplicate_documents(documents: list[str]) -> list[str]:
    """
    Remove exact-duplicate documents, keeping the first occurrence of each.
    """
    seen_hashes = set()
    unique_documents = []

    for doc in documents:
        fingerprint = hash_text(doc)
        if fingerprint not in seen_hashes:
            seen_hashes.add(fingerprint)
            unique_documents.append(doc)

    return unique_documents