from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", encoding="utf-8-sig")

SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "replace-me")
DEBUG = os.getenv("DJANGO_DEBUG", "true").lower() == "true"
ALLOWED_HOSTS = [host.strip() for host in os.getenv("DJANGO_ALLOWED_HOSTS", "*").split(",") if host.strip()]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "corsheaders",
    "rest_framework",
    "rest_framework.authtoken",
    "django_filters",
    "drf_spectacular",
    "apps.users",
    "apps.ventures",
    "apps.knowledge",
    "apps.advisors",
    "apps.evaluations",
]

MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"

DATABASES = {
    "default": {
        "ENGINE": os.getenv("DB_ENGINE", "django.db.backends.sqlite3"),
        "NAME": os.getenv("DB_NAME", str(BASE_DIR / "db.sqlite3")),
        "USER": os.getenv("DB_USER", ""),
        "PASSWORD": os.getenv("DB_PASSWORD", ""),
        "HOST": os.getenv("DB_HOST", ""),
        "PORT": os.getenv("DB_PORT", ""),
    }
}

AUTH_USER_MODEL = "users.User"

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = os.getenv("DJANGO_TIME_ZONE", "Europe/Berlin")
USE_I18N = True
USE_TZ = True

STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

CORS_ALLOW_ALL_ORIGINS = os.getenv("CORS_ALLOW_ALL_ORIGINS", "true").lower() == "true"

REST_FRAMEWORK = {
    "DEFAULT_SCHEMA_CLASS": "drf_spectacular.openapi.AutoSchema",
    "DEFAULT_FILTER_BACKENDS": ["django_filters.rest_framework.DjangoFilterBackend"],
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework.authentication.SessionAuthentication",
        "rest_framework.authentication.TokenAuthentication",
    ],
}

SPECTACULAR_SETTINGS = {
    "TITLE": "Founder Assistant API",
    "DESCRIPTION": "LLM-assisted startup support system backend",
    "VERSION": "0.1.0",
}

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

LOCAL_LLM_PROVIDER = os.getenv("LOCAL_LLM_PROVIDER", "ollama")
LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:11434")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "qwen2.5:7b")
LOCAL_LLM_TIMEOUT_SECONDS = int(os.getenv("LOCAL_LLM_TIMEOUT_SECONDS", "360"))
LOCAL_LLM_API_KEY = os.getenv("LOCAL_LLM_API_KEY", "")
LOCAL_LLM_VERIFY_SSL = os.getenv("LOCAL_LLM_VERIFY_SSL", "true").lower() not in {"0", "false", "no", "off"}
LOCAL_RAG_EMBED_MODEL = os.getenv("LOCAL_RAG_EMBED_MODEL", "bge-m3")
LOCAL_RAG_RERANKER_MODEL = os.getenv("LOCAL_RAG_RERANKER_MODEL", "")
LOCAL_RAG_RERANKER_REVISION = os.getenv("LOCAL_RAG_RERANKER_REVISION", "")
LOCAL_RAG_RERANKER_DEVICE = os.getenv("LOCAL_RAG_RERANKER_DEVICE", "auto")
LOCAL_RAG_RERANKER_BATCH_SIZE = int(os.getenv("LOCAL_RAG_RERANKER_BATCH_SIZE", "4"))
LOCAL_RAG_RERANKER_MAX_LENGTH = int(os.getenv("LOCAL_RAG_RERANKER_MAX_LENGTH", "512"))
LOCAL_RAG_DATA_DIR = os.getenv("LOCAL_RAG_DATA_DIR", str(BASE_DIR.parent / "knowledge_base"))
LOCAL_RAG_INDEX_DIR = os.getenv("LOCAL_RAG_INDEX_DIR", str(BASE_DIR / "rag_index"))
LOCAL_RAG_CHUNK_SIZE = int(os.getenv("LOCAL_RAG_CHUNK_SIZE", "1200"))
LOCAL_RAG_CHUNK_OVERLAP = int(os.getenv("LOCAL_RAG_CHUNK_OVERLAP", "180"))
LOCAL_RAG_MIN_CHUNK_CHARS = int(os.getenv("LOCAL_RAG_MIN_CHUNK_CHARS", "180"))

# OPENAI_API_KEY remains in the process environment so it cannot be serialized
# accidentally through Django settings snapshots or evaluation artifacts.
OPENAI_JUDGE_MODEL = os.getenv("OPENAI_JUDGE_MODEL", "gpt-5.4-mini")
OPENAI_JUDGE_REASONING_EFFORT = os.getenv(
    "OPENAI_JUDGE_REASONING_EFFORT",
    "low",
)
OPENAI_JUDGE_MAX_OUTPUT_TOKENS = int(
    os.getenv("OPENAI_JUDGE_MAX_OUTPUT_TOKENS", "2500")
)
OPENAI_JUDGE_TIMEOUT_SECONDS = int(os.getenv("OPENAI_JUDGE_TIMEOUT_SECONDS", "120"))
OPENAI_JUDGE_MAX_RETRIES = int(os.getenv("OPENAI_JUDGE_MAX_RETRIES", "2"))

# DEEPSEEK_API_KEY likewise remains only in the process environment. DeepSeek's
# Responses-compatible endpoint allows reuse of the same blinded Judge pipeline.
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_JUDGE_MODEL = os.getenv("DEEPSEEK_JUDGE_MODEL", "deepseek-v4-flash")
DEEPSEEK_JUDGE_REASONING_EFFORT = os.getenv(
    "DEEPSEEK_JUDGE_REASONING_EFFORT",
    "high",
)
DEEPSEEK_JUDGE_MAX_OUTPUT_TOKENS = int(
    os.getenv("DEEPSEEK_JUDGE_MAX_OUTPUT_TOKENS", "8000")
)
DEEPSEEK_JUDGE_TIMEOUT_SECONDS = int(
    os.getenv("DEEPSEEK_JUDGE_TIMEOUT_SECONDS", "360")
)
DEEPSEEK_JUDGE_MAX_RETRIES = int(os.getenv("DEEPSEEK_JUDGE_MAX_RETRIES", "3"))
