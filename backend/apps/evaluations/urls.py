"""URL routing for benchmark endpoint (demo-minimal mode)."""

from django.urls import path

from .views import BenchmarkRunView, BenchmarkTestSetView

urlpatterns = [
    path("benchmark/test-set/", BenchmarkTestSetView.as_view(), name="benchmark-test-set"),
    path("benchmark/run/", BenchmarkRunView.as_view(), name="benchmark-run"),
]
