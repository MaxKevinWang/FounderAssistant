"""Model profile definitions used by baseline-vs-RAG experiments.

This module intentionally keeps model metadata in one place so that:
1) evaluation scripts can report exactly which model stack was used,
2) UI/docs can display consistent model naming,
3) future model switching does not require touching business logic files.
"""

from dataclasses import dataclass

# Shared requested output budget for RAG answers and answer-generation benchmarks.
# Explicit request/run settings may override it for provider-specific needs.
DEFAULT_ANSWER_MAX_TOKENS = 900


@dataclass(frozen=True)
class ModelProfile:
    """Declarative profile for one model pipeline variant.

    Attributes:
        name: Internal identifier used in code and experiment outputs.
        llm_model: Generator model used for answer generation.
        embedding_model: Embedding model used for dense retrieval (RAG only).
        reranker_model: Optional reranker model for second-stage ranking.
        use_rag: Whether retrieval augmentation is enabled.
    """

    name: str
    llm_model: str
    embedding_model: str | None = None
    reranker_model: str | None = None
    use_rag: bool = False


@dataclass(frozen=True)
class RAGRuntimeProfile:
    """Runtime defaults for one generation-model family in consumer RAG mode.

    `top_k` controls how many retrieved chunks are injected into the final
    generation prompt. `max_tokens` is the output budget requested from the model
    service. Both families use the same defaults; the family label is retained
    for response metadata and evaluation reports.
    """

    model_type: str
    top_k: int
    max_tokens: int


# Models whose GGUF metadata exposes `expert_count` / `expert_used_count`.
# These are MoE-style generation models available through the institute WebUI.
EXPERT_GENERATION_MODELS = {
    "cloud.moonshot/kimi-k3",
    "gpt-oss:120b",
    "gpt-oss:20b",
    "gemma4:26b",
    "kimi-k3",
    "moonshotai/Kimi-K3",
    "nemotron-3-nano:latest",
    "nemotron-3-super:latest",
    "qwen3.6:latest",
}


DENSE_RAG_RUNTIME_PROFILE = RAGRuntimeProfile(
    model_type="dense",
    # Keep retrieval width and requested output budget independent of family.
    top_k=5,
    max_tokens=DEFAULT_ANSWER_MAX_TOKENS,
)


EXPERT_RAG_RUNTIME_PROFILE = RAGRuntimeProfile(
    model_type="expert",
    # Use the same answer-generation defaults as the dense profile.
    top_k=5,
    max_tokens=DEFAULT_ANSWER_MAX_TOKENS,
)


def classify_generation_model(model_name: str | None) -> str:
    """Return `expert` for known MoE models; otherwise default to `dense`."""

    normalized = (model_name or "").strip()
    if normalized in EXPERT_GENERATION_MODELS:
        return "expert"
    return "dense"


def resolve_rag_runtime_profile(model_name: str | None) -> RAGRuntimeProfile:
    """Choose RAG defaults from the selected generation model type."""

    if classify_generation_model(model_name) == "expert":
        return EXPERT_RAG_RUNTIME_PROFILE
    return DENSE_RAG_RUNTIME_PROFILE


# Baseline profile keeps generation only (no retrieval pipeline).
# Keep the generator model identical to the RAG profile so retrieval is the
# main experimental variable in baseline-vs-RAG comparisons.
BASELINE_PROFILE = ModelProfile(
    name="baseline_qwen",
    llm_model="Qwen/Qwen2.5-14B-Instruct",
    use_rag=False,
)


# RAG profile keeps generator fixed for fair comparison and only adds retrieval stack.
# Embedding and reranker reflect a high-performing open-source retrieval stack.
RAG_PROFILE = ModelProfile(
    name="rag_qwen_bge",
    llm_model="Qwen/Qwen2.5-14B-Instruct",
    embedding_model="BAAI/bge-m3",
    reranker_model="BAAI/bge-reranker-v2-m3",
    use_rag=True,
)
