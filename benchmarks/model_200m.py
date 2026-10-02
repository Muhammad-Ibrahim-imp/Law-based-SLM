"""Factory and architecture checks for the benchmarked 199.4M model."""

from model.config import ModelConfig
from model.model import DecoderLM
from model.parameter_count import estimate_parameter_count

EXPECTED_PARAMETER_COUNT = 199_398_144


def build_model() -> DecoderLM:
    config = ModelConfig()
    estimated = estimate_parameter_count(config)["total"]
    if estimated != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"parameter estimate is {estimated:,}; "
            f"expected {EXPECTED_PARAMETER_COUNT:,}"
        )

    model = DecoderLM(config)
    actual = model.parameter_count
    if actual != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"model has {actual:,} parameters; "
            f"expected {EXPECTED_PARAMETER_COUNT:,}"
        )
    return model


if __name__ == "__main__":
    model = build_model()
    print(f"Exact parameter count: {model.parameter_count:,}")
