"""Text embedding backends for graph node features."""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

import config as project_config


PROJECT_ROOT = getattr(project_config, "PROJECT_ROOT", Path.cwd())
TEXT_EMBEDDING_BACKEND = getattr(project_config, "TEXT_EMBEDDING_BACKEND", "none")
SENTENCEBERT_MODEL_NAME = getattr(
    project_config,
    "SENTENCEBERT_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2",
)
SENTENCEBERT_EMBEDDING_DIM = getattr(project_config, "SENTENCEBERT_EMBEDDING_DIM", 384)
FINBERT_MODEL_NAME = getattr(project_config, "FINBERT_MODEL_NAME", "ProsusAI/finbert")
FINBERT_EMBEDDING_DIM = getattr(project_config, "FINBERT_EMBEDDING_DIM", 768)
TEXT_EMBEDDING_CACHE_DIR = getattr(
    project_config,
    "TEXT_EMBEDDING_CACHE_DIR",
    Path(PROJECT_ROOT) / "outputs" / "embedding_cache",
)
HF_MODEL_CACHE_DIR = getattr(project_config, "HF_MODEL_CACHE_DIR", Path(PROJECT_ROOT) / "outputs" / "hf_cache")
os.environ.setdefault("HF_HOME", str(HF_MODEL_CACHE_DIR))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(HF_MODEL_CACHE_DIR))


NO_TEXT_BACKENDS = {"", "none", "structural"}
TEXT_BACKENDS = {"sentencebert", "finbert", "sentencebert_finbert", *NO_TEXT_BACKENDS}


def normalize_embedding_backend(backend: str | None = None) -> str:
    value = (backend if backend is not None else TEXT_EMBEDDING_BACKEND).strip().lower()
    if value not in TEXT_BACKENDS:
        raise ValueError(f"Unsupported embedding backend: {value}")
    return "none" if value in NO_TEXT_BACKENDS else value


def text_embedding_dim(backend: str | None = None) -> int:
    normalized = normalize_embedding_backend(backend)
    if normalized == "sentencebert":
        return SENTENCEBERT_EMBEDDING_DIM
    if normalized == "finbert":
        return FINBERT_EMBEDDING_DIM
    if normalized == "sentencebert_finbert":
        return SENTENCEBERT_EMBEDDING_DIM + FINBERT_EMBEDDING_DIM
    return 0


def encode_texts(texts: list[str], backend: str | None = None) -> list[list[float]]:
    normalized = normalize_embedding_backend(backend)
    if normalized == "none":
        return [[] for _text in texts]
    cached = [_read_cache(text, normalized) for text in texts]
    missing_indices = [index for index, value in enumerate(cached) if value is None]
    if missing_indices:
        missing_texts = [texts[index] for index in missing_indices]
        encoded = _encode_uncached(missing_texts, normalized)
        for index, vector in zip(missing_indices, encoded):
            cached[index] = vector
            _write_cache(texts[index], normalized, vector)
    return [_fit_dim(vector or [], text_embedding_dim(normalized)) for vector in cached]


def _encode_uncached(texts: list[str], backend: str) -> list[list[float]]:
    if backend == "sentencebert":
        return _encode_sentencebert(texts)
    if backend == "finbert":
        return _encode_finbert(texts)
    if backend == "sentencebert_finbert":
        sentence_vectors = _encode_sentencebert(texts)
        finbert_vectors = _encode_finbert(texts)
        return [sentence + finbert for sentence, finbert in zip(sentence_vectors, finbert_vectors)]
    raise ValueError(f"Unsupported embedding backend: {backend}")


def _encode_sentencebert(texts: list[str]) -> list[list[float]]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "SentenceBERT backend requires sentence-transformers. "
            "Install optional embedding dependencies before using --embedding-backend sentencebert."
        ) from exc
    model = _sentencebert_model(SentenceTransformer)
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return [_fit_dim(vector.tolist(), SENTENCEBERT_EMBEDDING_DIM) for vector in vectors]


def _encode_finbert(texts: list[str]) -> list[list[float]]:
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "FinBERT backend requires transformers. "
            "Install optional embedding dependencies before using --embedding-backend finbert."
        ) from exc
    tokenizer, model = _finbert_model(AutoTokenizer, AutoModel)
    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    with torch.no_grad():
        output = model(**batch)
    hidden = output.last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    pooled = F.normalize(pooled, p=2, dim=1)
    return [_fit_dim(vector.tolist(), FINBERT_EMBEDDING_DIM) for vector in pooled]


@lru_cache(maxsize=1)
def _sentencebert_model(sentence_transformer_cls):
    return sentence_transformer_cls(
        SENTENCEBERT_MODEL_NAME,
        cache_folder=str(HF_MODEL_CACHE_DIR),
        local_files_only=_has_hf_cache(SENTENCEBERT_MODEL_NAME),
    )


@lru_cache(maxsize=1)
def _finbert_model(tokenizer_cls, model_cls):
    local_only = _has_hf_cache(FINBERT_MODEL_NAME)
    tokenizer = tokenizer_cls.from_pretrained(
        FINBERT_MODEL_NAME,
        cache_dir=str(HF_MODEL_CACHE_DIR),
        local_files_only=local_only,
    )
    model = model_cls.from_pretrained(
        FINBERT_MODEL_NAME,
        cache_dir=str(HF_MODEL_CACHE_DIR),
        local_files_only=local_only,
    )
    model.eval()
    return tokenizer, model


def _has_hf_cache(model_name: str) -> bool:
    cache_name = "models--" + model_name.replace("/", "--")
    return (Path(HF_MODEL_CACHE_DIR) / cache_name).exists()


def _fit_dim(vector: list[float], dim: int) -> list[float]:
    if len(vector) == dim:
        return [float(value) for value in vector]
    if len(vector) > dim:
        return [float(value) for value in vector[:dim]]
    return [float(value) for value in vector] + [0.0] * (dim - len(vector))


def _read_cache(text: str, backend: str) -> list[float] | None:
    path = _cache_path(text, backend)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    vector = data.get("vector") if isinstance(data, dict) else None
    if not isinstance(vector, list):
        return None
    return [float(value) for value in vector]


def _write_cache(text: str, backend: str, vector: list[float]) -> None:
    path = _cache_path(text, backend)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"backend": backend, "vector": vector}), encoding="utf-8")
    except OSError:
        return


def _cache_path(text: str, backend: str) -> Path:
    payload: dict[str, Any] = {
        "backend": backend,
        "sentencebert_model": SENTENCEBERT_MODEL_NAME,
        "finbert_model": FINBERT_MODEL_NAME,
        "text": text,
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return Path(TEXT_EMBEDDING_CACHE_DIR) / backend / f"{digest}.json"
