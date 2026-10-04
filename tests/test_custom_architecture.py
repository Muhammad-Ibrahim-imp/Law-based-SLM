from architectures.custom import CustomConfig, DecoderLM
from architectures.registry import build_model


def _raises_not_implemented(fn, match: str) -> None:
    try:
        fn()
    except NotImplementedError as exc:
        assert match in str(exc)
        return
    raise AssertionError("expected NotImplementedError")


def test_registry_rejects_custom_until_implemented():
    _raises_not_implemented(
        lambda: build_model("custom"),
        "architectures/custom.py",
    )


def test_direct_construction_is_also_unimplemented():
    _raises_not_implemented(
        lambda: DecoderLM(CustomConfig()),
        "not implemented",
    )
