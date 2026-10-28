"""HTTP API layer for demo-focused advisor endpoints (LLM + RAG only)."""

import json

from django.http import StreamingHttpResponse
from rest_framework.renderers import BaseRenderer, JSONRenderer
from rest_framework import serializers, status
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from .local_llm import LocalLLMError, check_status, generate_text
from .model_profiles import resolve_rag_runtime_profile
from .request_guard import assess_advisor_request
from .rag_pipeline import (
    RAGError,
    build_rag_index,
    get_rag_status,
    list_rag_chunks,
    rag_answer,
    rag_answer_stream,
    update_rag_chunk_metadata,
)


# These are the three curated collections used in the evaluated RAG pipeline.
# Public chat may also omit the domain explicitly for an all-domain search.
SUPPORTED_RAG_DECISION_DOMAINS = (
    "legal_form_selection",
    "funding_preparation",
    "market_validation",
)
MAX_CONVERSATION_HISTORY_MESSAGES = 12


class LocalLLMChatRequestSerializer(serializers.Serializer):
    """Validation schema for baseline local-model generation endpoint."""

    prompt = serializers.CharField(max_length=4000)
    system_prompt = serializers.CharField(required=False, allow_blank=True, default="")
    temperature = serializers.FloatField(required=False, min_value=0, max_value=2, default=0.2)
    max_tokens = serializers.IntegerField(required=False, min_value=32, max_value=4096, default=512)
    model = serializers.CharField(required=False, allow_blank=False, max_length=128)


class LocalLLMStatusView(APIView):
    """Health/status endpoint for local Ollama generator availability."""

    permission_classes = [AllowAny]

    def get(self, request):
        """Return model reachability and installed-model metadata."""

        try:
            return Response(check_status(), status=status.HTTP_200_OK)
        except LocalLLMError as exc:
            # Keep 200 with reachable=false so UI can render a recoverable status panel.
            return Response(
                {
                    "reachable": False,
                    "error": str(exc),
                },
                status=status.HTTP_200_OK,
            )


class LocalLLMChatView(APIView):
    """Baseline answer endpoint (no retrieval augmentation)."""

    permission_classes = [AllowAny]

    def post(self, request):
        """Validate request and generate one response from local model."""

        serializer = LocalLLMChatRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        data = serializer.validated_data
        try:
            result = generate_text(
                prompt=data["prompt"],
                system_prompt=data["system_prompt"],
                temperature=data["temperature"],
                max_tokens=data["max_tokens"],
                model=data.get("model"),
            )
            return Response(result, status=status.HTTP_200_OK)
        except LocalLLMError as exc:
            return Response(
                {"error": str(exc)},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )


class RAGBuildRequestSerializer(serializers.Serializer):
    """Optional build overrides for index generation."""

    data_dir = serializers.CharField(required=False, allow_blank=False)
    index_dir = serializers.CharField(required=False, allow_blank=False)
    embedding_model = serializers.CharField(required=False, allow_blank=False)
    chunk_size = serializers.IntegerField(required=False, min_value=200, max_value=4000)
    chunk_overlap = serializers.IntegerField(required=False, min_value=0, max_value=1000)
    min_chunk_chars = serializers.IntegerField(required=False, min_value=80, max_value=2000)


class RAGConversationMessageSerializer(serializers.Serializer):
    """One untrusted user/assistant message used only as bounded context."""

    role = serializers.ChoiceField(choices=("user", "assistant"))
    content = serializers.CharField(max_length=4000, trim_whitespace=True)


class RAGChatRequestSerializer(serializers.Serializer):
    """Validation schema for scoped or explicit all-domain RAG requests."""

    prompt = serializers.CharField(max_length=4000)
    system_prompt = serializers.CharField(required=False, allow_blank=True, default="")
    temperature = serializers.FloatField(required=False, min_value=0, max_value=2, default=0.2)
    max_tokens = serializers.IntegerField(required=False, min_value=64, max_value=4096)
    # Production and evaluation both use a five-chunk dense context. Keep this
    # value fixed at the HTTP boundary; research ablations call rag_answer()
    # directly and preserve their settings in their run artifacts.
    top_k = serializers.IntegerField(required=False, min_value=5, max_value=5, default=5)
    domain = serializers.ChoiceField(
        required=False,
        allow_blank=True,
        default="",
        choices=SUPPORTED_RAG_DECISION_DOMAINS,
    )
    history = RAGConversationMessageSerializer(many=True, required=False, default=list)
    model = serializers.CharField(required=False, allow_blank=False, max_length=128)

    def validate_history(self, value):
        """Bound request size before the pipeline applies its token budget."""

        if len(value) > MAX_CONVERSATION_HISTORY_MESSAGES:
            raise serializers.ValidationError(
                f"History cannot contain more than {MAX_CONVERSATION_HISTORY_MESSAGES} messages."
            )
        return value


class AdvisorRequestAssessmentView(APIView):
    """Classify a consumer request before retrieval and answer generation."""

    permission_classes = [AllowAny]

    def post(self, request):
        serializer = RAGChatRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        payload = serializer.validated_data
        assessment = assess_advisor_request(
            payload["prompt"],
            domain=payload["domain"],
            conversation_history=payload["history"],
            model=payload.get("model"),
        )
        return Response(assessment.api_payload(), status=status.HTTP_200_OK)


class RAGChunkListRequestSerializer(serializers.Serializer):
    """Query parameters for expert chunk metadata review."""

    page = serializers.IntegerField(required=False, min_value=1, default=1)
    page_size = serializers.IntegerField(required=False, min_value=1, max_value=100, default=20)
    query = serializers.CharField(required=False, allow_blank=True, default="")
    domain = serializers.CharField(required=False, allow_blank=True, default="")
    reviewed = serializers.ChoiceField(
        required=False,
        allow_blank=True,
        choices=("", "true", "false"),
        default="",
    )


class RAGChunkMetadataUpdateSerializer(serializers.Serializer):
    """Patch payload for one chunk's editable metadata."""

    chunk_id = serializers.CharField(max_length=1000)
    metadata = serializers.DictField(child=serializers.JSONField(), allow_empty=False)


class RAGStatusView(APIView):
    """Return current RAG readiness (index + model availability)."""

    permission_classes = [AllowAny]

    def get(self, request):
        """Expose `rag_pipeline.get_rag_status` as HTTP endpoint."""

        return Response(get_rag_status(), status=status.HTTP_200_OK)


class RAGBuildIndexView(APIView):
    """Trigger index build from documents under configured data directory."""

    permission_classes = [AllowAny]

    def post(self, request):
        """Validate optional overrides and execute index build pipeline."""

        serializer = RAGBuildRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        payload = serializer.validated_data

        try:
            result = build_rag_index(payload if payload else None)
            return Response(result, status=status.HTTP_200_OK)
        except RAGError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)


class RAGChunkListView(APIView):
    """Paginated expert view over chunk metadata."""

    permission_classes = [AllowAny]

    def get(self, request):
        """Return chunk metadata rows with optional filtering."""

        serializer = RAGChunkListRequestSerializer(data=request.query_params)
        serializer.is_valid(raise_exception=True)
        payload = serializer.validated_data
        try:
            return Response(
                list_rag_chunks(
                    page=payload["page"],
                    page_size=payload["page_size"],
                    query=payload["query"],
                    domain=payload["domain"],
                    reviewed=payload["reviewed"],
                ),
                status=status.HTTP_200_OK,
            )
        except RAGError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)


class RAGChunkMetadataUpdateView(APIView):
    """Persist manual metadata edits for one chunk."""

    permission_classes = [AllowAny]

    def patch(self, request):
        """Update editable metadata fields for a chunk."""

        serializer = RAGChunkMetadataUpdateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        payload = serializer.validated_data
        try:
            return Response(
                update_rag_chunk_metadata(
                    chunk_id=payload["chunk_id"],
                    metadata_patch=payload["metadata"],
                ),
                status=status.HTTP_200_OK,
            )
        except RAGError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)


class RAGChatView(APIView):
    """Answer endpoint that uses retrieval + grounded generation."""

    permission_classes = [AllowAny]

    def post(self, request):
        """Run retrieval, construct grounded prompt, then generate final answer."""

        serializer = RAGChatRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        payload = serializer.validated_data
        requested_model = payload.get("model")
        runtime_profile = resolve_rag_runtime_profile(requested_model)
        resolved_top_k = runtime_profile.top_k
        resolved_max_tokens = payload.get("max_tokens", runtime_profile.max_tokens)

        try:
            result = rag_answer(
                question=payload["prompt"],
                top_k=resolved_top_k,
                domain=payload["domain"],
                system_prompt=payload["system_prompt"],
                temperature=payload["temperature"],
                max_tokens=resolved_max_tokens,
                model=requested_model,
                conversation_history=payload["history"],
            )
            result["runtime_profile"] = {
                "model_type": runtime_profile.model_type,
                "top_k": resolved_top_k,
                "max_tokens": resolved_max_tokens,
                "auto_applied": "max_tokens" not in request.data,
                "domain_filter_applied": bool(payload["domain"]),
                **({"domain": payload["domain"]} if payload["domain"] else {}),
            }
            return Response(result, status=status.HTTP_200_OK)
        except (RAGError, LocalLLMError) as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)


def _sse_event(event_name: str, payload: dict) -> str:
    """Serialize one Server-Sent Event frame."""

    return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


class EventStreamRenderer(BaseRenderer):
    """Renderer marker so DRF accepts `Accept: text/event-stream` requests."""

    media_type = "text/event-stream"
    format = "event-stream"
    charset = "utf-8"

    def render(self, data, accepted_media_type=None, renderer_context=None):
        if data is None:
            return b""
        if isinstance(data, bytes):
            return data
        return json.dumps(data, ensure_ascii=False).encode("utf-8")


class RAGChatStreamView(APIView):
    """Streaming RAG endpoint for consumer chat interfaces."""

    permission_classes = [AllowAny]
    renderer_classes = [EventStreamRenderer, JSONRenderer]

    def post(self, request):
        """Validate request and stream retrieval metadata + generation tokens."""

        serializer = RAGChatRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        payload = serializer.validated_data
        requested_model = payload.get("model")
        runtime_profile = resolve_rag_runtime_profile(requested_model)
        resolved_top_k = runtime_profile.top_k
        resolved_max_tokens = payload.get("max_tokens", runtime_profile.max_tokens)

        def event_stream():
            try:
                yield _sse_event(
                    "profile",
                    {
                        "model_type": runtime_profile.model_type,
                        "top_k": resolved_top_k,
                        "max_tokens": resolved_max_tokens,
                        "auto_applied": "max_tokens" not in request.data,
                        "domain_filter_applied": bool(payload["domain"]),
                        **({"domain": payload["domain"]} if payload["domain"] else {}),
                    },
                )
                for event in rag_answer_stream(
                    question=payload["prompt"],
                    top_k=resolved_top_k,
                    domain=payload["domain"],
                    system_prompt=payload["system_prompt"],
                    temperature=payload["temperature"],
                    max_tokens=resolved_max_tokens,
                    model=requested_model,
                    conversation_history=payload["history"],
                ):
                    event_type = event.pop("type", "message")
                    yield _sse_event(event_type, event)
            except (RAGError, LocalLLMError) as exc:
                yield _sse_event("error", {"error": str(exc)})

        response = StreamingHttpResponse(event_stream(), content_type="text/event-stream")
        response["Cache-Control"] = "no-cache"
        response["X-Accel-Buffering"] = "no"
        return response
