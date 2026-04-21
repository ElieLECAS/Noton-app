"""
Module central d'instrumentation LangSmith.

Usage :
    from app.tracing import trace_run

    with trace_run("vector_retrieval", run_type="retriever", inputs={"query": q}) as run:
        results = do_retrieval(q)
        run.end(outputs={"nb_results": len(results)})

Si LANGCHAIN_TRACING_V2 est false ou LANGSMITH_API_KEY absent, toutes les fonctions
deviennent des no-ops : aucun import langsmith ne se produit, pas d'erreur.
"""
from __future__ import annotations

import os
import logging
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

logger = logging.getLogger(__name__)

_TRACING_ENABLED: bool = False
_CLIENT = None


def init_langsmith() -> None:
    """
    Configure le SDK LangSmith à partir des variables d'environnement / settings.
    Doit être appelé une fois au démarrage de l'app (main.py) et du worker Celery.
    Idempotent.
    """
    global _TRACING_ENABLED, _CLIENT

    api_key = os.environ.get("LANGSMITH_API_KEY", "").strip()
    tracing_raw = os.environ.get("LANGCHAIN_TRACING_V2", "false").strip().lower()
    tracing_enabled = tracing_raw in ("true", "1", "yes")
    project = os.environ.get("LANGCHAIN_PROJECT", "noton-rag-kag").strip()

    if not api_key or not tracing_enabled:
        logger.info(
            "LangSmith désactivé (LANGCHAIN_TRACING_V2=%s, clé présente=%s)",
            tracing_raw,
            bool(api_key),
        )
        return

    try:
        import langsmith  # noqa: F401

        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGSMITH_API_KEY", api_key)
        os.environ.setdefault("LANGCHAIN_PROJECT", project)

        from langsmith import Client

        _CLIENT = Client(api_key=api_key)
        _TRACING_ENABLED = True
        logger.info("✅ LangSmith activé — projet=%s", project)
    except ImportError:
        logger.warning(
            "langsmith non installé (pip install langsmith). Tracing désactivé."
        )
    except Exception as exc:
        logger.warning("LangSmith init échoué (%s) — tracing désactivé.", exc)


def is_tracing_enabled() -> bool:
    return _TRACING_ENABLED


class _NoOpRun:
    """Remplace un vrai run LangSmith quand le tracing est désactivé."""

    def end(self, outputs: Optional[Dict[str, Any]] = None, error: Optional[str] = None) -> None:
        pass

    def add_metadata(self, metadata: Dict[str, Any]) -> None:
        pass


class _LangSmithRun:
    """Wrapper autour d'un run LangSmith (langsmith.run_trees.RunTree)."""

    def __init__(self, run_tree: Any) -> None:
        self._run = run_tree

    def end(self, outputs: Optional[Dict[str, Any]] = None, error: Optional[str] = None) -> None:
        try:
            if error:
                self._run.end(error=error)
            else:
                self._run.end(outputs=outputs or {})
            self._run.patch()
        except Exception as exc:
            logger.debug("LangSmith run.end() échoué : %s", exc)

    def add_metadata(self, metadata: Dict[str, Any]) -> None:
        try:
            self._run.extra = {**(self._run.extra or {}), "metadata": metadata}
        except Exception:
            pass


class TraceRunContext:
    def __init__(
        self,
        name: str,
        run_type: str = "chain",
        inputs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[list] = None,
        parent_run_id: Optional[str] = None,
    ):
        self.name = name
        self.run_type = run_type
        self.inputs = inputs
        self.metadata = metadata
        self.tags = tags
        self.parent_run_id = parent_run_id
        self.wrapper = None

    def __enter__(self) -> _NoOpRun | _LangSmithRun:
        if not _TRACING_ENABLED:
            return _NoOpRun()

        try:
            from langsmith.run_trees import RunTree

            run = RunTree(
                name=self.name,
                run_type=self.run_type,
                inputs=self.inputs or {},
                extra={"metadata": self.metadata or {}},
                tags=self.tags or [],
                parent_run_id=self.parent_run_id,
            )
            run.post()
            self.wrapper = _LangSmithRun(run)
            return self.wrapper
        except ImportError:
            return _NoOpRun()
        except Exception as exc:
            logger.debug("trace_run setup échoué (%s), no-op.", exc)
            return _NoOpRun()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.wrapper:
            if exc_type:
                self.wrapper.end(error=str(exc_val))
            else:
                self.wrapper.end()


def trace_run(
    name: str,
    run_type: str = "chain",
    inputs: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[list] = None,
    parent_run_id: Optional[str] = None,
) -> TraceRunContext:
    """
    Ouvre un run LangSmith (span) et le ferme automatiquement.
    Version robuste basée sur une classe pour éviter "generator didn't stop after throw()".
    """
    return TraceRunContext(
        name=name,
        run_type=run_type,
        inputs=inputs,
        metadata=metadata,
        tags=tags,
        parent_run_id=parent_run_id,
    )


class TracePipelineContext:
    def __init__(
        self,
        name: str,
        inputs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[list] = None,
    ):
        self.name = name
        self.inputs = inputs
        self.metadata = metadata
        self.tags = tags
        self.inner_context = None

    def __enter__(self) -> _NoOpRun | _LangSmithRun:
        self.inner_context = trace_run(
            name=self.name,
            run_type="chain",
            inputs=self.inputs,
            metadata=self.metadata,
            tags=self.tags,
        )
        return self.inner_context.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.inner_context:
            return self.inner_context.__exit__(exc_type, exc_val, exc_tb)


def trace_pipeline(
    name: str,
    inputs: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[list] = None,
) -> TracePipelineContext:
    """
    Raccourci pour ouvrir une trace root de type 'chain'.
    Version robuste basée sur une classe.
    """
    return TracePipelineContext(
        name=name,
        inputs=inputs,
        metadata=metadata,
        tags=tags,
    )
