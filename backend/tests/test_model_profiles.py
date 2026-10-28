"""Tests for generation-model classification and shared answer defaults."""

import inspect

import pytest

from apps.advisors.model_profiles import (
    DEFAULT_ANSWER_MAX_TOKENS,
    classify_generation_model,
    resolve_rag_runtime_profile,
)


def test_kimi_k3_gateway_and_public_ids_are_expert_models() -> None:
    """Classify every Kimi K3 identifier used by the project as MoE/expert."""

    assert classify_generation_model("cloud.moonshot/kimi-k3") == "expert"
    assert classify_generation_model("kimi-k3") == "expert"
    assert classify_generation_model("moonshotai/Kimi-K3") == "expert"


def test_unknown_generation_model_still_defaults_to_dense() -> None:
    """Preserve the existing conservative fallback for unregistered models."""

    assert classify_generation_model("unknown:model") == "dense"


def test_all_model_families_use_top_five_retrieval() -> None:
    """Keep retrieved context constant across dense and expert comparisons."""

    assert resolve_rag_runtime_profile("unknown:model").top_k == 5
    assert resolve_rag_runtime_profile("cloud.moonshot/kimi-k3").top_k == 5


@pytest.mark.parametrize("model", [
    "gemma4:31b", "gemma4:26b", "gpt-oss:20b", "magistral:latest",
    "olmo-3.1:32b-think", "qwen3.6:latest", "cloud.moonshot/kimi-k3",
    "unknown:model",
])
def test_model_families_share_answer_budget(model: str) -> None:
    assert DEFAULT_ANSWER_MAX_TOKENS == 900
    assert resolve_rag_runtime_profile(model).max_tokens == DEFAULT_ANSWER_MAX_TOKENS


def test_answer_generation_entry_points_share_default_budget() -> None:
    from apps.advisors.rag_pipeline import rag_answer, rag_answer_stream
    from apps.evaluations.benchmarking import (
        run_baseline_rag_benchmark, run_baseline_rag_benchmark_batch,
    )
    from apps.evaluations.formal_development_benchmark import run_formal_development_benchmark
    from apps.evaluations.management.commands.run_baseline_rag_eval import Command as ScenarioCommand
    from apps.evaluations.management.commands.run_formal_development_eval import Command as FormalCommand
    from apps.evaluations.views import BenchmarkRunRequestSerializer

    for function, parameter in [
        (rag_answer, "max_tokens"),
        (rag_answer_stream, "max_tokens"),
        (run_formal_development_benchmark, "max_tokens"),
        (run_baseline_rag_benchmark, "max_tokens_scenario"),
        (run_baseline_rag_benchmark_batch, "max_tokens_scenario"),
    ]:
        assert inspect.signature(function).parameters[parameter].default == DEFAULT_ANSWER_MAX_TOKENS

    for command, parameter, option in [
        (FormalCommand(), "max_tokens", "--max-tokens"),
        (ScenarioCommand(), "max_tokens_scenario", "--max-tokens-scenario"),
    ]:
        parser = command.create_parser("manage.py", "test")
        assert vars(parser.parse_args([]))[parameter] == DEFAULT_ANSWER_MAX_TOKENS
        assert vars(parser.parse_args([option, "8192"]))[parameter] == 8192

    serializer = BenchmarkRunRequestSerializer(data={})
    assert serializer.is_valid(), serializer.errors
    assert serializer.validated_data["max_tokens_scenario"] == DEFAULT_ANSWER_MAX_TOKENS
