"""Embeddings: pluggable providers with an honest offline fallback.

The dedupe path needs vectors. A dense model gives the best evidence, but the
engine must still run with no key, no download, and no network. So this module
exposes four embedders behind one protocol, prefers a **local** sentence
transformer, and **says out loud** which one it picked. A degraded run is never
silent.

Preference order: explicit override, then :class:`LocalEmbedder`, then a paid
provider when its key exists, then :class:`LexicalEmbedder`.

Every vector is cached on disk keyed ``(embedder_name, sha256(text))`` under
``ORACLE_HOME/embeddings/``, so a second ``domain add`` costs nothing.

No subject matter lives in this file.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
from collections import Counter
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path

    from the_oracle.config import Settings

log = logging.getLogger(__name__)

#: Every lexical embedder name starts with this. Callers use it to detect the
#: degraded mode and tighten their evidence bar.
LEXICAL_PREFIX = "lexical"

#: Character n-gram sizes used by the offline fallback.
NGRAM_SIZES: tuple[int, ...] = (3, 4, 5)

#: Hashed feature space for the fallback. Large enough that collisions are rare
#: for short objective texts, small enough to stay cheap.
LEXICAL_DIM = 2048

_WORD_RE = re.compile(r"[a-z0-9]+")


class LocalEmbedderUnavailableError(RuntimeError):
    """Raised when the local sentence-transformers model cannot be loaded."""


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns text into unit-length vectors."""

    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per input text, in input order."""
        ...


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def sha256_text(text: str) -> str:
    """Stable digest of one text, the cache key half that is not the name."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def l2_normalise(vector: list[float]) -> list[float]:
    """Scale to unit length. An all-zero vector is returned unchanged."""
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0.0:
        return vector
    return [x / norm for x in vector]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity, clamped to [-1, 1]."""
    if len(a) != len(b):
        msg = f"vector length mismatch: {len(a)} != {len(b)}"
        raise ValueError(msg)
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


def normalise_text(text: str) -> str:
    """Case-folded, whitespace-collapsed text. Deterministic."""
    return " ".join(_WORD_RE.findall(text.casefold()))


# ---------------------------------------------------------------------------
# offline fallback
# ---------------------------------------------------------------------------


class LexicalEmbedder:
    """Character n-gram TF-IDF, hashed, L2-normalised. No network, no key.

    Term frequency is sublinear (``1 + log tf``). Inverse document frequency is
    optional: unfitted the embedder weights every n-gram equally, which keeps it
    a pure function of one text and therefore safe to cache. Calling
    :meth:`fit` folds a corpus in and changes :attr:`name`, so fitted vectors
    can never collide with unfitted ones in the cache.
    """

    def __init__(self, dim: int = LEXICAL_DIM, sizes: Sequence[int] = NGRAM_SIZES) -> None:
        self.dim = int(dim)
        self.sizes = tuple(sizes)
        self._idf: dict[str, float] = {}
        self._fit_tag = ""
        self.name = self._make_name()

    def _make_name(self) -> str:
        sizes = "".join(str(s) for s in self.sizes)
        suffix = f"-{self._fit_tag}" if self._fit_tag else ""
        return f"{LEXICAL_PREFIX}-char{sizes}-tfidf-d{self.dim}{suffix}"

    # -- features ----------------------------------------------------------

    def ngrams(self, text: str) -> list[str]:
        """Padded character n-grams over the normalised text."""
        clean = f" {normalise_text(text)} "
        grams: list[str] = []
        for size in self.sizes:
            if len(clean) < size:
                continue
            grams.extend(clean[i : i + size] for i in range(len(clean) - size + 1))
        return grams

    def fit(self, corpus: Sequence[str]) -> "LexicalEmbedder":
        """Learn document frequencies from ``corpus``. Optional and explicit."""
        docs = [set(self.ngrams(t)) for t in corpus]
        total = len(docs) or 1
        counts: Counter[str] = Counter()
        for doc in docs:
            counts.update(doc)
        self._idf = {
            gram: math.log((total + 1.0) / (df + 1.0)) + 1.0 for gram, df in counts.items()
        }
        digest = hashlib.sha256(
            "\n".join(sorted(normalise_text(t) for t in corpus)).encode("utf-8")
        ).hexdigest()[:12]
        self._fit_tag = f"fit{digest}"
        self.name = self._make_name()
        return self

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        counts: Counter[str] = Counter(self.ngrams(text))
        vector = [0.0] * self.dim
        for gram, count in counts.items():
            weight = (1.0 + math.log(count)) * self._idf.get(gram, 1.0)
            index = int(hashlib.blake2b(gram.encode("utf-8"), digest_size=8).hexdigest(), 16)
            vector[index % self.dim] += weight
        return l2_normalise(vector)


# ---------------------------------------------------------------------------
# local dense model (preferred)
# ---------------------------------------------------------------------------

#: Small, fast, strong on short-text similarity. 384 dimensions.
DEFAULT_LOCAL_MODEL = "BAAI/bge-small-en-v1.5"


class LocalEmbedder:
    """A sentence-transformers model running on this machine. Free and offline.

    ``sentence_transformers`` is imported inside the constructor, so the rest of
    the engine never pays for it and never needs it installed. Vectors are
    L2-normalised, so cosine similarity is a plain dot product.
    """

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id or os.environ.get("ORACLE_EMBED_MODEL") or DEFAULT_LOCAL_MODEL
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on the machine
            msg = (
                "sentence-transformers is not installed; "
                "install it to use the local embedder"
            )
            raise LocalEmbedderUnavailableError(msg) from exc
        self._model = SentenceTransformer(self.model_id)
        self.name = f"local:{self.model_id}"
        # sentence-transformers 6.x renamed this; support both without pinning.
        dim_fn = getattr(
            self._model,
            "get_embedding_dimension",
            None,
        ) or self._model.get_sentence_embedding_dimension
        self.dim = int(dim_fn() or 0)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        raw = self._model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return [l2_normalise([float(x) for x in row]) for row in raw]


def local_available() -> bool:
    """True when ``sentence_transformers`` can be imported. No model download."""
    import importlib.util

    return importlib.util.find_spec("sentence_transformers") is not None


# ---------------------------------------------------------------------------
# real providers
# ---------------------------------------------------------------------------


class _HTTPEmbedder:
    """Shared plumbing for HTTP embedding providers."""

    url: str = ""
    env_key: str = ""

    def __init__(self, model: str, dim: int, api_key: str | None = None) -> None:
        self.model = model
        self.dim = dim
        self.name = f"{self.provider}:{model}"
        self._api_key = api_key or os.environ.get(self.env_key, "")
        if not self._api_key:
            msg = f"{type(self).__name__} needs {self.env_key}"
            raise RuntimeError(msg)

    @property
    def provider(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def _payload(self, texts: Sequence[str]) -> dict[str, object]:  # pragma: no cover
        raise NotImplementedError

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        import httpx  # local import: the offline path must not need it

        response = httpx.post(
            self.url, json=self._payload(texts), headers=self._headers(), timeout=60.0
        )
        response.raise_for_status()
        rows = response.json()["data"]
        rows = sorted(rows, key=lambda r: r.get("index", 0))
        return [l2_normalise([float(x) for x in row["embedding"]]) for row in rows]


class OpenAIEmbedder(_HTTPEmbedder):
    """OpenAI embeddings. Used only when ``OPENAI_API_KEY`` is set."""

    url = "https://api.openai.com/v1/embeddings"
    env_key = "OPENAI_API_KEY"

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dim: int = 1536,
        api_key: str | None = None,
    ) -> None:
        super().__init__(model, dim, api_key)

    @property
    def provider(self) -> str:
        return "openai"

    def _payload(self, texts: Sequence[str]) -> dict[str, object]:
        return {"model": self.model, "input": list(texts)}


class VoyageEmbedder(_HTTPEmbedder):
    """Voyage embeddings. Used only when ``VOYAGE_API_KEY`` is set."""

    url = "https://api.voyageai.com/v1/embeddings"
    env_key = "VOYAGE_API_KEY"

    def __init__(
        self, model: str = "voyage-3", dim: int = 1024, api_key: str | None = None
    ) -> None:
        super().__init__(model, dim, api_key)

    @property
    def provider(self) -> str:
        return "voyage"

    def _payload(self, texts: Sequence[str]) -> dict[str, object]:
        return {"model": self.model, "input": list(texts), "input_type": "document"}


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------


def cache_dir() -> "Path":
    """``ORACLE_HOME/embeddings``. Created lazily by the cache itself."""
    from the_oracle.domains.registry import oracle_home

    return oracle_home() / "embeddings"


_SAFE = re.compile(r"[^a-z0-9._-]+")


def _slug(name: str) -> str:
    return _SAFE.sub("-", name.casefold()).strip("-") or "embedder"


class CachedEmbedder:
    """Wrap an embedder with a ``(name, sha256(text))`` disk cache.

    Delegates :attr:`name` and :attr:`dim` so callers cannot tell the wrapper
    apart from the real thing - including the lexical-mode check.
    """

    def __init__(self, inner: Embedder, directory: "Path | None" = None) -> None:
        self.inner = inner
        self.name = inner.name
        self.dim = inner.dim
        self._dir_override = directory
        self._memory: dict[str, list[float]] = {}

    @property
    def directory(self) -> "Path":
        base = self._dir_override or cache_dir()
        return base / _slug(self.name)

    def _path(self, digest: str) -> "Path":
        return self.directory / f"{digest}.json"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float] | None] = [None] * len(texts)
        missing: list[int] = []
        digests = [sha256_text(t) for t in texts]
        for i, digest in enumerate(digests):
            hit = self._memory.get(digest)
            if hit is None:
                hit = self._read(digest)
            if hit is None:
                missing.append(i)
            else:
                self._memory[digest] = hit
                out[i] = hit
        if missing:
            fresh = self.inner.embed([texts[i] for i in missing])
            for i, vector in zip(missing, fresh, strict=True):
                self._memory[digests[i]] = vector
                self._write(digests[i], vector)
                out[i] = vector
        return [v for v in out if v is not None]

    def _read(self, digest: str) -> list[float] | None:
        path = self._path(digest)
        if not path.is_file():
            return None
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if body.get("embedder") != self.name:
            return None
        return [float(x) for x in body.get("vector", [])]

    def _write(self, digest: str, vector: list[float]) -> None:
        path = self._path(digest)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"embedder": self.name, "sha256": digest, "vector": vector}),
                encoding="utf-8",
            )
        except OSError as exc:  # pragma: no cover - a cache miss is not fatal
            log.warning("embedding cache write failed at %s: %s", path, exc)


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------


def is_lexical(embedder: Embedder | None) -> bool:
    """True when the vectors come from the offline fallback, not a provider."""
    if embedder is None:
        return True
    return str(getattr(embedder, "name", "")).startswith(LEXICAL_PREFIX)


def _route(settings: "Settings | None") -> tuple[str, str]:
    """The configured embed route, split into ``(provider, model)``."""
    from the_oracle.config import Task, get_settings

    active = settings or get_settings()
    route = active.model_for(Task.EMBED)
    provider, _, model = route.partition(":")
    return provider, (model or route)


def _override() -> tuple[str, str]:
    """``ORACLE_EMBEDDER`` as ``(kind, model)``. Empty kind means no override.

    Accepted forms: ``local``, ``local:<model id>``, ``openai``,
    ``openai:<model>``, ``voyage``, ``voyage:<model>``, ``lexical``.
    """
    raw = os.environ.get("ORACLE_EMBEDDER", "").strip()
    kind, _, model = raw.partition(":")
    return kind.casefold(), model


def _build(kind: str, model: str, route_model: str) -> Embedder:
    """Construct one embedder by kind. Raises if that kind is unavailable."""
    if kind == "local":
        return LocalEmbedder(model or None)
    if kind == "openai":
        return OpenAIEmbedder(model or route_model or "text-embedding-3-small")
    if kind == "voyage":
        return VoyageEmbedder(model or route_model or "voyage-3")
    if kind == "lexical":
        return LexicalEmbedder()
    msg = f"unknown embedder kind {kind!r}"
    raise ValueError(msg)


def get_embedder(settings: "Settings | None" = None, *, cache: bool | None = None) -> Embedder:
    """Pick an embedder and log the choice.

    Order: explicit ``ORACLE_EMBEDDER`` override, then the local dense model,
    then a keyed provider, then the lexical fallback. The fallback logs a
    warning that names the consequence, because dedupe quality drops with it.
    """
    provider, route_model = _route(settings)
    chosen: Embedder | None = None

    kind, model = _override()
    if kind:
        chosen = _build(kind, model, route_model)
        log.info("embedder: %s (explicit ORACLE_EMBEDDER override)", chosen.name)

    if chosen is None and local_available():
        try:
            chosen = LocalEmbedder()
        except (LocalEmbedderUnavailableError, OSError, ValueError) as exc:
            log.warning("local embedder unavailable (%s); trying a provider key", exc)

    if chosen is None:
        if provider == "openai" and os.environ.get("OPENAI_API_KEY"):
            chosen = OpenAIEmbedder(route_model)
        elif provider == "voyage" and os.environ.get("VOYAGE_API_KEY"):
            chosen = VoyageEmbedder(route_model)
        elif os.environ.get("OPENAI_API_KEY"):
            chosen = OpenAIEmbedder()
        elif os.environ.get("VOYAGE_API_KEY"):
            chosen = VoyageEmbedder()

    if chosen is None:
        chosen = LexicalEmbedder()
        log.warning(
            "no dense embedder available (sentence-transformers absent, no "
            "provider key); falling back to %s. Similarity is lexical only, so "
            "dedupe thresholds are raised and reuse becomes rarer. Install "
            "sentence-transformers or set OPENAI_API_KEY / VOYAGE_API_KEY for "
            "real semantic matching.",
            chosen.name,
        )
    else:
        log.info("embedder: %s (dim %d)", chosen.name, chosen.dim)

    use_cache = cache
    if use_cache is None:
        use_cache = os.environ.get("ORACLE_EMBED_CACHE", "1") not in ("0", "false", "")
    return CachedEmbedder(chosen) if use_cache else chosen


def describe(embedder: Embedder) -> str:
    """One short line naming the embedder and whether it is degraded."""
    mode = "LEXICAL FALLBACK (degraded)" if is_lexical(embedder) else "provider"
    return f"{embedder.name} dim={embedder.dim} [{mode}]"
