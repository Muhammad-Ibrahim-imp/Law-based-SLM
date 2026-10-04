"""Print total, embedding, and non-embedding parameter counts.

Usage:
    python -m scripts.count_architecture_params
"""

from architectures.registry import build_model, implemented_names


def main() -> None:
    for name in implemented_names():
        model = build_model(name)
        total = model.parameter_count
        embedding = model.embedding_parameter_count
        print(
            f"{name}: total={total:,} "
            f"embedding={embedding:,} "
            f"non_embedding={total - embedding:,}"
        )


if __name__ == "__main__":
    main()
