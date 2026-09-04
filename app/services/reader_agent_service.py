"""Lecteur agentique — boucle d'outils bornée autour de la génération.

Plan ``docs/plan_lecteur_agentique_2026-09-02.md`` (phase 2b). Principe : le pipeline
déterministe (compréhension, périmètre, retrieval, élection) produit un **pack initial
court** ; ensuite le générateur (Mistral Large) décide lui-même ce qu'il lit, dans un
espace d'actions fermé (les outils de ``reader_tools``), sous budget, avec journal.

Ce module ne connaît ni la base ni les outils concrets : il reçoit une fonction de stream
(celle du routeur, pour que les tests qui la patchent restent valides), une liste de
``ToolSpec`` et un budget, et produit des événements SSE prêts à émettre :
``stage`` (une action = une étape visible), ``thinking`` (la phrase de plan du modèle
avant chaque appel), ``message`` (la réponse finale, émise après le contrôle de sortie).

Garanties de l'orchestrateur — que les outils ne peuvent pas assurer eux-mêmes :
  * budget : ≤ N appels, deadline, images cumulées ; le compteur restant figure dans chaque
    résultat d'outil, sinon le modèle le brûle ou ne s'en sert pas ;
  * arrêt sur non-progrès : un appel identique (même outil, mêmes arguments) répété force
    la réponse au round suivant (``tool_choice="none"``) ;
  * élagage des images : la limite de 8 images est PAR REQUÊTE et chaque round renvoie
    tout l'historique ; les images des rounds antérieurs sont remplacées par un placeholder
    texte — sans cela le 3ᵉ round échoue en 400 (code 3051) ;
  * images et messages ``tool`` : un message ``tool`` ne porte que du texte ; les PNG d'un
    outil partent dans un message ``user`` juste après, légendés ;
  * contrôle de sortie programmatique sur pack ∪ résultats d'outils : un KO repart comme
    message de contrôle et le lecteur relit (une fois), au lieu d'une régénération aveugle ;
  * repli : une panne d'infrastructure d'outils (400) relance UNE fois sans outils sur le
    pack initial — même code, tracé ``degraded``, jamais un flag.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional, Set, Tuple

import httpx

from app.services.context_packer_service import MISTRAL_MAX_IMAGES_PER_REQUEST
from app.services.stream_source_filter import (
    EvidenceTagStreamFilter,
    SourcesTagStreamFilter,
    chain_filters,
    finalize_filters,
)

logger = logging.getLogger(__name__)

# Taille des morceaux du replay « machine à écrire » de la réponse finale.
_REPLAY_CHUNK = 60
# Sous ce reliquat de deadline, on n'ouvre plus de round d'outils.
_MIN_SECONDS_FOR_TOOLS = 5.0


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class ToolResult:
    """Ce qu'un outil rend à l'orchestrateur."""

    text: str
    # {"b64": str, "document_id": int, "page_no": int, "document_title": str, "label": str}
    images: List[Dict[str, Any]] = field(default_factory=list)
    # Pages réellement LUES (texte ou image) : alimentent les sources et le registre.
    pages_read: List[Tuple[int, int]] = field(default_factory=list)
    # Documents rencontrés : id → titre (registre des documents lus).
    documents: Dict[int, str] = field(default_factory=dict)
    # Texte à joindre au corpus de preuve (défaut : ``text``). Vide = rien (ex. image seule).
    evidence: Optional[str] = None
    error: bool = False


ToolHandler = Callable[[Dict[str, Any]], Awaitable[ToolResult]]
LabelFn = Callable[[Dict[str, Any]], str]


@dataclass
class ToolSpec:
    name: str
    schema: Dict[str, Any]  # {"type": "function", "function": {...}} (format Mistral)
    handler: ToolHandler
    max_images: int = 0  # plafond d'images rendues par appel
    label: Optional[LabelFn] = None  # libellé de l'étape visible (événement ``stage``)


@dataclass
class LoopBudget:
    max_tool_calls: int = 6
    deadline_s: float = 60.0
    max_tool_images: int = 8  # images fournies par les outils, cumulées sur le tour


@dataclass
class ReadDocument:
    document_id: int
    index: int
    title: str
    pages: Set[int] = field(default_factory=set)
    has_source_file: bool = True
    base: Dict[str, Any] = field(default_factory=dict)  # entrée cag_documents d'origine


StreamFn = Callable[..., AsyncIterator[str]]
# (réponse, corpus de preuve texte, citations <evidence>, pages vues en IMAGE) → verdict.
# Les pages vues en image sont nécessaires au contrôle : il ne lit que du texte, et une cote
# portée sur une coupe n'y figure jamais.
OutputCheck = Callable[[str, str, List[str], List[Tuple[int, int]]], Dict[str, Any]]


def sse(obj: Dict[str, Any]) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def _parse_args(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        data = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _call_key(name: str, args: Dict[str, Any]) -> str:
    try:
        return name + ":" + json.dumps(args, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        return name + ":" + repr(args)


# ---------------------------------------------------------------------------
# La boucle
# ---------------------------------------------------------------------------


class ReaderLoop:
    """Une instance par tour. ``run()`` est un générateur d'événements SSE ; l'état final
    (texte, sources, preuves, journal, vérification) se lit sur l'instance ensuite."""

    def __init__(
        self,
        *,
        messages: List[Dict[str, Any]],
        model: str,
        stream_fn: StreamFn,
        tools: List[ToolSpec],
        budget: Optional[LoopBudget] = None,
        max_tokens: Optional[int] = None,
        initial_documents: Optional[List[Dict[str, Any]]] = None,
        evidence_seed: Optional[List[str]] = None,
        initial_images_meta: Optional[List[Dict[str, Any]]] = None,
        output_check: Optional[OutputCheck] = None,
        max_control_rounds: int = 1,
        replay_final: bool = True,
    ) -> None:
        self.messages = messages
        self._base_len = len(messages)
        self.model = model
        self.stream_fn = stream_fn
        self.tools = list(tools or [])
        self._specs = {t.name: t for t in self.tools}
        self.budget = budget or LoopBudget()
        self.max_tokens = max_tokens
        self.output_check = output_check
        self.max_control_rounds = max(0, int(max_control_rounds))
        self.replay_final = replay_final

        # Registre des documents lus (pack initial en tête, outils ensuite).
        self.read_documents: Dict[int, ReadDocument] = {}
        for d in initial_documents or []:
            did = d.get("document_id")
            if did is None:
                continue
            self.read_documents[int(did)] = ReadDocument(
                document_id=int(did),
                index=int(d.get("index") or (len(self.read_documents) + 1)),
                title=str(d.get("document_title") or "Document"),
                pages={int(p) for p in (d.get("pages") or []) if isinstance(p, int)},
                has_source_file=bool(d.get("has_source_file", True)),
                base=dict(d),
            )
        # Légendes des images du message initial (pour l'élagage).
        self._initial_images_meta = list(initial_images_meta or [])
        # Pages dont le lecteur a VU l'image (pack initial + outils) : le contrôle de sortie
        # ne lit que du texte et doit savoir ce qu'il ne peut pas vérifier.
        self.image_pages_seen: Set[Tuple[int, int]] = {
            (int(m["document_id"]), int(m["page_no"]))
            for m in self._initial_images_meta
            if m.get("document_id") is not None and m.get("page_no") is not None
        }

        # Corpus de preuve : ce que le lecteur a réellement lu.
        self.evidence_parts: List[str] = [b for b in (evidence_seed or []) if b]

        # État de la boucle.
        self.tool_calls_used = 0
        self.tool_images_used = 0
        self._seen_calls: Set[str] = set()
        self._stop_tools = False
        self.degraded = False
        self.stopped_by: Optional[str] = None

        # Sorties.
        self.final_text = ""
        self.final_raw_text = ""
        self.source_filter: Optional[SourcesTagStreamFilter] = None
        self.evidence_citations: List[str] = []
        self.verification: Optional[Dict[str, Any]] = None
        self.reasoning_parts: List[str] = []
        self.trace: Dict[str, Any] = {
            "rounds": [],
            "tool_calls": 0,
            "tool_images": 0,
            "control_rounds": 0,
            "degraded": None,
            "stopped_by": None,
            "duration_ms": None,
            "tools_available": [t.name for t in self.tools],
        }

    # ------------------------------------------------------------------
    # Propriétés
    # ------------------------------------------------------------------

    @property
    def evidence_text(self) -> str:
        return "\n\n".join(self.evidence_parts)

    @property
    def used_documents(self) -> List[Dict[str, Any]]:
        return list(self.source_filter.used_documents) if self.source_filter else []

    def used_pages_by_index(self) -> Dict[int, List[int]]:
        """Pages citées dans ``<sources>`` par INDEX de document lu (clé ``doc`` ou ``doc_id``)."""
        by_id = {d.document_id: d.index for d in self.read_documents.values()}
        result: Dict[int, List[int]] = {}
        for item in self.used_documents:
            idx: Optional[int] = None
            if item.get("doc_id") is not None and int(item["doc_id"]) in by_id:
                idx = by_id[int(item["doc_id"])]
            elif item.get("doc") is not None:
                candidate = int(item["doc"])
                # « doc » peut désigner un index de pack… ou un id de document si le modèle
                # a confondu les deux : on tranche par ce qui existe réellement.
                if any(d.index == candidate for d in self.read_documents.values()):
                    idx = candidate
                elif candidate in by_id:
                    idx = by_id[candidate]
            if idx is None:
                continue
            result.setdefault(idx, []).extend(int(p) for p in item.get("pages") or [])
        return result

    def documents_for_sources(self) -> List[Dict[str, Any]]:
        """Entrées au format ``cag_documents`` (pack + lectures d'outils) pour les sources UI."""
        out: List[Dict[str, Any]] = []
        for d in sorted(self.read_documents.values(), key=lambda x: x.index):
            entry = dict(d.base)
            entry.update(
                {
                    "index": d.index,
                    "document_id": d.document_id,
                    "document_title": d.title,
                    "pages": sorted(d.pages),
                    "has_source_file": d.has_source_file,
                }
            )
            entry.setdefault("matched_pages", [])
            entry.setdefault("seed_pages", [])
            entry.setdefault("full_document", False)
            entry.setdefault("score", 0.0)
            entry.setdefault("election_score", 0.0)
            out.append(entry)
        return out

    # ------------------------------------------------------------------
    # Exécution
    # ------------------------------------------------------------------

    async def run(self) -> AsyncIterator[str]:
        t0 = time.perf_counter()
        tools_enabled = bool(self.tools)
        control_rounds = 0
        try:
            while True:
                round_no = len(self.trace["rounds"]) + 1
                elapsed = time.perf_counter() - t0
                remaining_calls = self.budget.max_tool_calls - self.tool_calls_used
                allow_tools = (
                    tools_enabled
                    and remaining_calls > 0
                    and (self.budget.deadline_s - elapsed) > _MIN_SECONDS_FOR_TOOLS
                    and not self._stop_tools
                )
                if tools_enabled and not allow_tools:
                    self.stopped_by = self.stopped_by or (
                        "deadline" if (self.budget.deadline_s - elapsed) <= _MIN_SECONDS_FOR_TOOLS
                        else ("repeat" if self._stop_tools else "budget")
                    )
                self._prune_images()

                kwargs: Dict[str, Any] = {}
                if tools_enabled:
                    kwargs["tools"] = [t.schema for t in self.tools]
                    kwargs["tool_choice"] = "auto" if allow_tools else "none"

                text_parts: List[str] = []
                tool_calls: List[Dict[str, Any]] = []
                round_t0 = time.perf_counter()
                try:
                    async for raw in self.stream_fn(
                        "", model=self.model, context=self.messages, max_tokens=self.max_tokens, **kwargs
                    ):
                        try:
                            parsed = json.loads(raw)
                        except (TypeError, json.JSONDecodeError):
                            continue
                        thinking = parsed.get("thinking")
                        if thinking:
                            self.reasoning_parts.append(thinking)
                            yield sse({"thinking": thinking})
                            continue
                        if parsed.get("tool_calls"):
                            tool_calls = list(parsed["tool_calls"])
                            continue
                        content = (parsed.get("message") or {}).get("content") or ""
                        if content:
                            text_parts.append(content)
                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code if exc.response is not None else None
                    if tools_enabled and status == 400 and not text_parts:
                        body = ""
                        try:
                            body = exc.response.text[:300] if exc.response is not None else ""
                        except Exception:  # noqa: BLE001
                            body = ""
                        logger.warning(
                            "[lecteur] 400 avec outils (%s) — repli SANS outils sur le pack initial",
                            body,
                        )
                        tools_enabled = False
                        self.degraded = True
                        self.trace["degraded"] = f"tools_unavailable: {body}"
                        self._strip_tool_messages()
                        continue
                    raise

                text = "".join(text_parts)

                # ——— Round d'outils ———
                if tool_calls:
                    plan = text.strip()
                    if plan:
                        self.reasoning_parts.append(plan)
                        yield sse({"thinking": plan})
                    self.messages.append(
                        {"role": "assistant", "content": text, "tool_calls": tool_calls}
                    )
                    round_entry: Dict[str, Any] = {
                        "round": round_no,
                        "plan": plan[:600],
                        "calls": [],
                        "llm_ms": int((time.perf_counter() - round_t0) * 1000),
                    }
                    for stage_label in self._stage_labels(tool_calls):
                        yield sse({"stage": {"key": "tool", "label": stage_label}})
                    results = await self._execute(tool_calls, round_entry)
                    images_batch: List[Dict[str, Any]] = []
                    for tc, res in zip(tool_calls, results):
                        self.messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.get("id"),
                                "name": (tc.get("function") or {}).get("name"),
                                "content": res.text,
                            }
                        )
                        images_batch.extend(res.images)
                    if images_batch:
                        self.messages.append(self._images_message(images_batch))
                    self.trace["rounds"].append(round_entry)
                    continue

                # ——— Réponse finale ———
                self.final_raw_text = text
                display = self._filter_final(text)
                self.final_text = display
                self.trace["rounds"].append(
                    {
                        "round": round_no,
                        "plan": "",
                        "calls": [],
                        "final": True,
                        "llm_ms": int((time.perf_counter() - round_t0) * 1000),
                    }
                )

                if self.output_check is not None:
                    try:
                        verdict = self.output_check(
                            display,
                            self.evidence_text,
                            self.evidence_citations,
                            sorted(self.image_pages_seen),
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("[lecteur] contrôle de sortie en échec (%s) — réponse émise", exc)
                        verdict = None
                    if verdict is not None:
                        previous = self.verification
                        self.verification = verdict
                        if (
                            not verdict.get("ok")
                            and verdict.get("feedback")
                            and tools_enabled
                            and control_rounds < self.max_control_rounds
                            and (self.budget.deadline_s - (time.perf_counter() - t0)) > _MIN_SECONDS_FOR_TOOLS
                        ):
                            control_rounds += 1
                            self.trace["control_rounds"] = control_rounds
                            # Le brouillon reste dans l'historique du tour (pas dans la
                            # conversation persistée) : le lecteur corrige EN RELISANT.
                            self.messages.append({"role": "assistant", "content": text})
                            # role="system" (et non "user") : sinon le modèle traite ce tour
                            # comme un vrai message de l'utilisateur et lui répond directement
                            # ("vous avez raison...") au lieu de corriger en silence.
                            self.messages.append({"role": "system", "content": verdict["feedback"]})
                            # Un round de contrôle rouvre les outils même après un arrêt
                            # « répétition » : la consigne change la question posée.
                            self._stop_tools = False
                            if self.tool_calls_used >= self.budget.max_tool_calls:
                                self.budget.max_tool_calls = self.tool_calls_used + 2
                            yield sse({"stage": {"key": "control", "label": "Contrôle des références"}})
                            continue
                        if control_rounds and previous is not None:
                            verdict["repaired"] = True
                            verdict["unsupported_before_repair"] = list(
                                previous.get("unsupported_claims") or []
                            )
                            verdict["unsupported_after_repair"] = list(
                                verdict.get("unsupported_claims") or []
                            )
                        verdict["action"] = (
                            "passed" if verdict.get("ok")
                            else ("flagged" if not control_rounds else "flagged")
                        )
                        if verdict.get("ok") and control_rounds:
                            verdict["action"] = "repaired"

                if self.replay_final:
                    for i in range(0, len(display), _REPLAY_CHUNK):
                        yield sse({"message": {"content": display[i : i + _REPLAY_CHUNK]}})
                self.stopped_by = self.stopped_by or "final"
                break
        finally:
            self.trace["tool_calls"] = self.tool_calls_used
            self.trace["tool_images"] = self.tool_images_used
            self.trace["stopped_by"] = self.stopped_by
            self.trace["duration_ms"] = int((time.perf_counter() - t0) * 1000)
            if self.verification is not None:
                self.trace["verification_action"] = self.verification.get("action")

    # ------------------------------------------------------------------
    # Outils
    # ------------------------------------------------------------------

    def _stage_labels(self, tool_calls: List[Dict[str, Any]]) -> List[str]:
        labels: List[str] = []
        for tc in tool_calls:
            fn = tc.get("function") or {}
            name = str(fn.get("name") or "")
            args = _parse_args(fn.get("arguments"))
            spec = self._specs.get(name)
            label = None
            if spec is not None and spec.label is not None:
                try:
                    label = spec.label(args)
                except Exception:  # noqa: BLE001
                    label = None
            labels.append(label or f"Outil {name or '?'}")
        return labels

    async def _execute(
        self, tool_calls: List[Dict[str, Any]], round_entry: Dict[str, Any]
    ) -> List[ToolResult]:
        """Exécute les appels d'un round (concurrents), applique budgets et registre."""
        planned: List[Tuple[int, Optional[ToolSpec], Dict[str, Any], Optional[ToolResult]]] = []
        for pos, tc in enumerate(tool_calls):
            fn = tc.get("function") or {}
            name = str(fn.get("name") or "")
            args = _parse_args(fn.get("arguments"))
            spec = self._specs.get(name)
            key = _call_key(name, args)
            precomputed: Optional[ToolResult] = None
            if spec is None:
                precomputed = ToolResult(
                    text=f"Outil inconnu : {name or '?'}. Outils disponibles : "
                    + ", ".join(self._specs) + ".",
                    error=True,
                )
            elif self.tool_calls_used >= self.budget.max_tool_calls:
                precomputed = ToolResult(
                    text="Budget d'appels épuisé : réponds maintenant avec ce que tu as lu ; "
                    "dis explicitement ce qui manque le cas échéant.",
                    error=True,
                )
                self._stop_tools = True
                self.stopped_by = self.stopped_by or "budget"
            elif key in self._seen_calls:
                precomputed = ToolResult(
                    text="Appel identique déjà effectué dans ce tour : le résultat serait le même. "
                    "Réponds avec ce que tu as lu, ou change de question / de document.",
                    error=True,
                )
                self._stop_tools = True
                self.stopped_by = self.stopped_by or "repeat"
            else:
                self._seen_calls.add(key)
                self.tool_calls_used += 1
            planned.append((pos, spec, args, precomputed))

        async def _run_one(spec: ToolSpec, args: Dict[str, Any]) -> ToolResult:
            t0 = time.perf_counter()
            try:
                res = await spec.handler(args)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[lecteur] outil %s en échec : %s", spec.name, exc)
                res = ToolResult(text=f"Erreur de l'outil {spec.name} : {exc}", error=True)
            res._ms = int((time.perf_counter() - t0) * 1000)  # type: ignore[attr-defined]
            return res

        coros = [
            _run_one(spec, args)
            for _, spec, args, pre in planned
            if pre is None and spec is not None
        ]
        executed = await asyncio.gather(*coros) if coros else []
        exec_iter = iter(executed)

        results: List[ToolResult] = []
        for pos, spec, args, pre in planned:
            res = pre if pre is not None else next(exec_iter)
            name = spec.name if spec is not None else str((tool_calls[pos].get("function") or {}).get("name"))
            # Plafonds d'images : par appel (spec) et cumulés (budget).
            if res.images:
                cap = spec.max_images if spec is not None else 0
                remaining_img = max(0, self.budget.max_tool_images - self.tool_images_used)
                keep = min(len(res.images), cap, remaining_img)
                dropped = len(res.images) - keep
                res.images = res.images[:keep]
                self.tool_images_used += keep
                if dropped > 0:
                    res.text += (
                        f"\n({dropped} image(s) non jointe(s) : plafond d'images atteint — "
                        "lis le texte, ou cible une seule page.)"
                    )
            # Registre des documents / pages lus + corpus de preuve.
            for did, title in (res.documents or {}).items():
                self._register_document(int(did), title)
            for did, page in res.pages_read or []:
                self._register_document(int(did), None)
                self.read_documents[int(did)].pages.add(int(page))
            for img in res.images or []:
                if img.get("document_id") is not None and img.get("page_no") is not None:
                    self.image_pages_seen.add((int(img["document_id"]), int(img["page_no"])))
            evidence = res.text if res.evidence is None else res.evidence
            if evidence and not res.error:
                self.evidence_parts.append(evidence)
            # Compteur de budget lisible par le modèle.
            res.text = (
                res.text.rstrip()
                + f"\n(appels restants : {max(0, self.budget.max_tool_calls - self.tool_calls_used)}"
                f"/{self.budget.max_tool_calls} · images restantes : "
                f"{max(0, self.budget.max_tool_images - self.tool_images_used)}/{self.budget.max_tool_images})"
            )
            round_entry["calls"].append(
                {
                    "tool": name,
                    "args": args,
                    "ms": getattr(res, "_ms", 0),
                    "chars": len(res.text),
                    "images": len(res.images),
                    "error": bool(res.error),
                    "pages_read": [list(p) for p in (res.pages_read or [])][:20],
                }
            )
            results.append(res)
        return results

    def _register_document(self, document_id: int, title: Optional[str]) -> None:
        doc = self.read_documents.get(document_id)
        if doc is None:
            next_index = max((d.index for d in self.read_documents.values()), default=0) + 1
            self.read_documents[document_id] = ReadDocument(
                document_id=document_id, index=next_index, title=title or f"Document {document_id}"
            )
        elif title and doc.title.startswith("Document "):
            doc.title = title

    # ------------------------------------------------------------------
    # Images
    # ------------------------------------------------------------------

    def _images_message(self, images: List[Dict[str, Any]]) -> Dict[str, Any]:
        lines = ["Images des pages demandées :"]
        meta: List[Dict[str, Any]] = []
        b64s: List[str] = []
        for i, img in enumerate(images, start=1):
            title = img.get("document_title") or f"Document {img.get('document_id')}"
            label = f" — {img['label']}" if img.get("label") else ""
            lines.append(
                f"Image {i} = document {img.get('document_id')} « {title} », page {img.get('page_no')}{label}"
            )
            meta.append(
                {
                    "document_id": img.get("document_id"),
                    "page_no": img.get("page_no"),
                    "document_title": title,
                    "label": img.get("label"),
                }
            )
            b64s.append(img["b64"])
        return {"role": "user", "content": "\n".join(lines), "images": b64s, "_reader_images_meta": meta}

    def _prune_images(self) -> None:
        """Ramène le total d'images de la requête sous le plafond API en remplaçant les
        images les plus ANCIENNES par un placeholder texte (le texte des pages, lui, est
        déjà dans l'historique via les résultats d'outils)."""
        total = sum(len(m.get("images") or []) for m in self.messages)
        if total <= MISTRAL_MAX_IMAGES_PER_REQUEST:
            return
        for pos, msg in enumerate(self.messages):
            imgs = msg.get("images") or []
            if not imgs:
                continue
            meta = msg.get("_reader_images_meta")
            if meta is None and pos < self._base_len:
                meta = self._initial_images_meta
            labels = []
            for i, _ in enumerate(imgs):
                m = meta[i] if meta and i < len(meta) else {}
                if m:
                    labels.append(f"doc {m.get('document_id')} p.{m.get('page_no')}")
                else:
                    labels.append(f"image {i + 1}")
            placeholder = (
                "\n[Images déjà vues, retirées de cette requête pour respecter la limite : "
                + ", ".join(labels)
                + " — leur texte extrait figure ci-dessus quand il existe ; redemande une "
                "image précise avec lire_pages ou zoomer si nécessaire.]"
            )
            msg["content"] = (str(msg.get("content") or "") + placeholder).strip()
            total -= len(imgs)
            msg["images"] = []
            if total <= MISTRAL_MAX_IMAGES_PER_REQUEST:
                break

    def _strip_tool_messages(self) -> None:
        """Repli sans outils : on repart du contexte initial (pack + question)."""
        del self.messages[self._base_len :]

    # ------------------------------------------------------------------
    # Sortie
    # ------------------------------------------------------------------

    def _filter_final(self, text: str) -> str:
        sf = SourcesTagStreamFilter()
        ef = EvidenceTagStreamFilter()
        filters = [sf, ef]
        out = chain_filters(text, filters)
        out += finalize_filters(filters)
        self.source_filter = sf
        self.evidence_citations = ef.citations
        return out.rstrip()
