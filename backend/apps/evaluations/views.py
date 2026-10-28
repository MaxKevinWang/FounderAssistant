"""HTTP API layer for benchmark execution (demo-minimal mode)."""

from pathlib import Path

from rest_framework import serializers, status
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.advisors.model_profiles import DEFAULT_ANSWER_MAX_TOKENS

from .benchmarking import (
    default_output_dir,
    default_test_set_path,
    load_test_set,
    run_baseline_rag_benchmark,
    run_baseline_rag_benchmark_batch,
)


class BenchmarkRunRequestSerializer(serializers.Serializer):
    """Validation schema for synchronous benchmark trigger payload."""

    mode = serializers.ChoiceField(choices=["both", "baseline", "rag"], default="both")
    tf_limit = serializers.IntegerField(required=False, min_value=1, max_value=200)
    repeats = serializers.IntegerField(required=False, min_value=1, max_value=20, default=1)
    skip_scenario = serializers.BooleanField(required=False, default=False)
    temperature = serializers.FloatField(required=False, min_value=0, max_value=2, default=0.1)
    top_k = serializers.IntegerField(required=False, min_value=1, max_value=12, default=5)
    max_tokens_tf = serializers.IntegerField(required=False, min_value=64, max_value=2048, default=220)
    max_tokens_scenario = serializers.IntegerField(
        required=False, min_value=128, max_value=4096, default=DEFAULT_ANSWER_MAX_TOKENS
    )
    generation_model = serializers.CharField(required=False, allow_blank=False, max_length=128)
    persist_db = serializers.BooleanField(required=False, default=False)
    owner_username = serializers.CharField(required=False, max_length=150, default="benchmark_runner")
    venture_name = serializers.CharField(required=False, max_length=200, default="Benchmark Venture")
    test_set = serializers.CharField(required=False, allow_blank=False)
    output_dir = serializers.CharField(required=False, allow_blank=False)


class BenchmarkTestSetView(APIView):
    """Return the full benchmark test-set payload for frontend rendering."""

    permission_classes = [AllowAny]

    def get(self, request):
        test_set_path = Path(request.query_params.get("test_set", str(default_test_set_path()))).resolve()
        try:
            payload = load_test_set(test_set_path)
        except Exception as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

        true_false = payload.get("true_false", [])
        user_scenarios = payload.get("user_scenarios", [])
        scenario = payload.get("scenario")
        if scenario and isinstance(scenario, dict):
            # Keep backward compatibility when `scenario` exists but user_scenarios is empty.
            if not user_scenarios:
                user_scenarios = [scenario]

        return Response(
            {
                "metadata": payload.get("metadata", {}),
                "true_false": true_false,
                "user_scenarios": user_scenarios,
                "count_true_false": len(true_false),
                "count_scenarios": len(user_scenarios),
            },
            status=status.HTTP_200_OK,
        )


class BenchmarkRunView(APIView):
    """Run benchmark in single or batch mode and return report payload."""

    permission_classes = [AllowAny]

    def post(self, request):
        serializer = BenchmarkRunRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        payload = serializer.validated_data

        try:
            repeats = int(payload.get("repeats", 1))
            if repeats > 1:
                batch = run_baseline_rag_benchmark_batch(
                    repeats=repeats,
                    test_set_path=Path(payload.get("test_set", str(default_test_set_path()))).resolve(),
                    output_dir=Path(payload.get("output_dir", str(default_output_dir()))).resolve(),
                    mode=str(payload.get("mode", "both")),
                    tf_limit=payload.get("tf_limit"),
                    skip_scenario=bool(payload.get("skip_scenario", False)),
                    temperature=float(payload.get("temperature", 0.1)),
                    top_k=int(payload.get("top_k", 5)),
                    max_tokens_tf=int(payload.get("max_tokens_tf", 220)),
                    max_tokens_scenario=int(payload.get("max_tokens_scenario", DEFAULT_ANSWER_MAX_TOKENS)),
                    generation_model=payload.get("generation_model"),
                    persist_db=bool(payload.get("persist_db", False)),
                    owner_username=str(payload.get("owner_username", "benchmark_runner")),
                    venture_name=str(payload.get("venture_name", "Benchmark Venture")),
                )
                return Response(
                    {
                        "is_batch": True,
                        "repeats": repeats,
                        "config": batch["batch_report"]["config"],
                        "batch_summary": batch["batch_report"]["aggregate"],
                        "runs": batch["batch_report"]["runs"],
                        "batch_json_path": batch["batch_json_path"],
                        "batch_md_path": batch["batch_md_path"],
                    },
                    status=status.HTTP_200_OK,
                )

            result = run_baseline_rag_benchmark(
                test_set_path=Path(payload.get("test_set", str(default_test_set_path()))).resolve(),
                output_dir=Path(payload.get("output_dir", str(default_output_dir()))).resolve(),
                mode=str(payload.get("mode", "both")),
                tf_limit=payload.get("tf_limit"),
                skip_scenario=bool(payload.get("skip_scenario", False)),
                temperature=float(payload.get("temperature", 0.1)),
                top_k=int(payload.get("top_k", 5)),
                max_tokens_tf=int(payload.get("max_tokens_tf", 220)),
                max_tokens_scenario=int(payload.get("max_tokens_scenario", DEFAULT_ANSWER_MAX_TOKENS)),
                generation_model=payload.get("generation_model"),
                persist_db=bool(payload.get("persist_db", False)),
                owner_username=str(payload.get("owner_username", "benchmark_runner")),
                venture_name=str(payload.get("venture_name", "Benchmark Venture")),
            )
        except Exception as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

        report = result["report"]
        return Response(
            {
                "is_batch": False,
                "summary": report["summary"],
                "config": report["config"],
                "tf_results": report["tf_results"],
                "scenario_result": report.get("scenario_result"),
                "json_path": result["json_path"],
                "md_path": result["md_path"],
                "db_persistence": result.get("db_persistence", {"persisted": False}),
            },
            status=status.HTTP_200_OK,
        )
