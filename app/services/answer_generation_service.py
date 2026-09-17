"""Génération de la réponse — un appel, un flux, un contrôle d'ancrage.

Le pipeline déterministe (compréhension, périmètre, retrieval, élection, packing) produit
le contexte ; ce module l'envoie au modèle et rend les événements SSE prêts à émettre :
``thinking`` (le raisonnement) et ``message`` (la réponse, streamée AU FIL DE L'EAU à
travers des filtres de balises incrémentaux).

Il n'y a **pas** de boucle d'outils. Le lecteur agentique a été mesuré inerte le 14/09 —
zéro appel d'outil sur les onze échecs de la campagne de référence — et le retirer n'a rien
coûté en justesse (83,9 % contre 82,3 %) tout en divisant par deux le délai de premier
token. Ses trois outils ne rendaient que du texte indexé déjà présent dans le pack : les
appeler n'apportait rien, et le modèle avait raison de s'en abstenir.

Ce qui subsiste ici est ce que l'orchestration assurait et qu'un appel nu n'assure pas :

  * **récupération des blocs machine égarés** — mesuré le 14/09, sur quatre tours des
    soixante-deux le modèle émet ``<sources>`` comme NOM d'outil ; sans rattrapage le bloc
    n'atteint jamais le filtre et les sources affichées retombent sur le repli (vingt pages
    au lieu d'une) ;
  * **recalage des pages citées** sur les pages réellement fournies — le modèle recopie le
    numéro IMPRIMÉ dans le cartouche de la planche au lieu du numéro de page du document ;
  * **contrôle d'ancrage programmatique** sur le corpus de preuve, dont le verdict est
    tracé et joint à la réponse.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Set, Tuple

from app.services.stream_source_filter import (
    EvidenceTagStreamFilter,
    SourcesTagStreamFilter,
    chain_filters,
    finalize_filters,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


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


# Un bloc machine de fin de réponse (``<sources>``, ``<evidence>``) parti dans le canal
# des appels d'outils au lieu du texte.
_MACHINE_BLOCK_RE = re.compile(r"<\s*/?\s*(?:sources|evidence)\s*>", re.IGNORECASE)
# Le modèle appelle parfois « sources » ou « evidence » comme un outil, sans les chevrons.
_MACHINE_BLOCK_NAMES = frozenset({"sources", "evidence"})


def split_leaked_blocks(
    tool_calls: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], str]:
    """Sépare les appels résiduels des blocs machine égarés dans ``tool_calls``.

    Mesuré le 14/09 : sur 4 tours des 62, le modèle a émis
    ``<sources>{"used":[{"doc_id":438,"pages":[5]}]}</sources>`` comme NOM d'outil, et le
    tour finissait avec une réponse VIDE ou des sources de repli. Aucun outil n'est plus
    proposé, mais le modèle continue d'utiliser ce canal de sa propre initiative : ce qu'il
    a voulu dire compte, pas le canal qu'il a choisi.

    Retourne (appels réels, texte récupéré à réinjecter dans la réponse).
    """
    reels: List[Dict[str, Any]] = []
    fuites: List[str] = []
    for call in tool_calls or []:
        fn = call.get("function") or {}
        name = str(fn.get("name") or "")
        raw_args = fn.get("arguments")
        args_text = raw_args if isinstance(raw_args, str) else ""
        nom_nu = name.strip().lower()
        if nom_nu in _MACHINE_BLOCK_NAMES:
            # Le modèle « appelle » sources / evidence comme s'il s'agissait d'un outil,
            # le bloc étant dans les arguments. On reconstitue la balise autour.
            fuites.append(f"<{nom_nu}>{args_text.strip()}</{nom_nu}>")
            continue
        if _MACHINE_BLOCK_RE.search(name) or _MACHINE_BLOCK_RE.search(args_text):
            morceau = name if _MACHINE_BLOCK_RE.search(name) else args_text
            fuites.append(morceau.strip())
            continue
        reels.append(call)
    return reels, "\n".join(f for f in fuites if f)


# ---------------------------------------------------------------------------
# La génération
# ---------------------------------------------------------------------------


class AnswerGeneration:
    """Une instance par tour. ``run()`` est un générateur d'événements SSE ; l'état final
    (texte, sources, preuves, trace, vérification) se lit sur l'instance ensuite."""

    def __init__(
        self,
        *,
        messages: List[Dict[str, Any]],
        model: str,
        stream_fn: StreamFn,
        max_tokens: Optional[int] = None,
        initial_documents: Optional[List[Dict[str, Any]]] = None,
        evidence_seed: Optional[List[str]] = None,
        initial_images_meta: Optional[List[Dict[str, Any]]] = None,
        output_check: Optional[OutputCheck] = None,
    ) -> None:
        self.messages = messages
        self.model = model
        self.stream_fn = stream_fn
        self.max_tokens = max_tokens
        self.output_check = output_check

        # Filtres de balises machine, INCRÉMENTAUX : ils sont montés avant le flux pour que
        # le texte parte au fil de l'eau (ordre imposé : sources puis evidence).
        self.source_filter: SourcesTagStreamFilter = SourcesTagStreamFilter()
        self._evidence_filter = EvidenceTagStreamFilter()
        self._filters = [self.source_filter, self._evidence_filter]

        # Registre des documents du pack.
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
        # Pages dont le modèle a VU l'image : le contrôle de sortie ne lit que du texte et
        # doit savoir ce qu'il ne peut pas vérifier.
        self._initial_images_meta = list(initial_images_meta or [])
        self.image_pages_seen: Set[Tuple[int, int]] = {
            (int(m["document_id"]), int(m["page_no"]))
            for m in self._initial_images_meta
            if m.get("document_id") is not None and m.get("page_no") is not None
        }

        # Corpus de preuve : les blocs documents du pack.
        self.evidence_parts: List[str] = [b for b in (evidence_seed or []) if b]

        # Sorties.
        self.final_text = ""
        self.final_raw_text = ""
        self.evidence_citations: List[str] = []
        self.verification: Optional[Dict[str, Any]] = None
        self.reasoning_parts: List[str] = []
        self.trace: Dict[str, Any] = {
            "duration_ms": None,
            "llm_ms": None,
            "first_token_ms": None,
            "leaked_blocks": 0,
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
        """Pages citées dans ``<sources>`` par INDEX de document (clé ``doc`` ou ``doc_id``).

        Les pages déclarées sont RECALÉES sur celles réellement packées. Mesuré le 14/09 :
        le modèle recopie le numéro IMPRIMÉ sur la planche (« Page 5 » dans le cartouche)
        au lieu du numéro de page du document (8) — décalage constant de −3 sur ce
        dossier, 51,6 % de citations vers la mauvaise page, et un clic sur la source qui
        ouvre autre chose que ce qui a servi à répondre. Le harness, lui, sait exactement
        quelles pages il a fournies : c'est cette liste qui fait foi.
        """
        by_id = {d.document_id: d.index for d in self.read_documents.values()}
        by_index = {d.index: d for d in self.read_documents.values()}
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
            declarees = [int(p) for p in item.get("pages") or []]
            lues = by_index[idx].pages if idx in by_index else set()
            if lues:
                retenues = [p for p in declarees if p in lues]
                if not retenues:
                    # Aucune page déclarée ne correspond à une page fournie : la
                    # déclaration est inexploitable (numéro imprimé, page inventée). On
                    # rend les pages RÉELLEMENT packées plutôt qu'un renvoi faux.
                    retenues = sorted(lues)
                    logger.info(
                        "[génération] sources recalées — doc %s : pages déclarées %s "
                        "introuvables parmi les pages fournies %s",
                        by_index[idx].document_id,
                        declarees,
                        retenues,
                    )
            else:
                retenues = declarees
            result.setdefault(idx, []).extend(retenues)
        return result

    def documents_for_sources(self) -> List[Dict[str, Any]]:
        """Entrées au format ``cag_documents`` pour les sources affichées."""
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
        """Streame la réponse AU FIL DE L'EAU, puis constate le verdict d'ancrage.

        Le texte part vers le client token par token : les filtres de balises sont
        incrémentaux et ne retiennent qu'un préfixe potentiel de ``<sources>`` /
        ``<evidence>``. Rien n'oblige plus à tamponner la réponse entière — la reprise
        après contrôle, seule chose qui pouvait la réécrire, a disparu avec la boucle
        d'outils. Le contrôle passe donc APRÈS l'affichage : il constate, il ne corrige pas.
        """
        t0 = time.perf_counter()
        premier_token_ms: Optional[int] = None
        try:
            emitted: List[str] = []
            raw_parts: List[str] = []
            leaked_calls: List[Dict[str, Any]] = []
            async for raw in self.stream_fn(
                "", model=self.model, context=self.messages, max_tokens=self.max_tokens
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
                    # Aucun outil n'est proposé : tout ce qui arrive par ce canal est un
                    # bloc machine égaré (cf. split_leaked_blocks).
                    leaked_calls.extend(parsed["tool_calls"])
                    continue
                content = (parsed.get("message") or {}).get("content") or ""
                if not content:
                    continue
                raw_parts.append(content)
                affichable = chain_filters(content, self._filters)
                if not affichable:
                    continue
                if premier_token_ms is None:
                    premier_token_ms = int((time.perf_counter() - t0) * 1000)
                emitted.append(affichable)
                yield sse({"message": {"content": affichable}})

            self.trace["llm_ms"] = int((time.perf_counter() - t0) * 1000)
            self.trace["first_token_ms"] = premier_token_ms

            # Le bloc machine égaré arrive par le canal des appels d'outils, donc APRÈS le
            # texte : on le fait traverser les mêmes filtres, qui le moissonnent sans rien
            # rendre d'affichable.
            if leaked_calls:
                _, fuite = split_leaked_blocks(leaked_calls)
                if fuite:
                    self.trace["leaked_blocks"] = len(leaked_calls)
                    logger.info(
                        "[génération] bloc machine récupéré du canal tool_calls (%d car.)",
                        len(fuite),
                    )
                    raw_parts.append(fuite)
                    residu = chain_filters(fuite, self._filters)
                    if residu:
                        emitted.append(residu)
                        yield sse({"message": {"content": residu}})

            # Fin de flux : ce que les filtres retenaient à tort (balise jamais complétée).
            tail = finalize_filters(self._filters)
            if tail:
                emitted.append(tail)
                yield sse({"message": {"content": tail}})

            self.final_raw_text = "".join(raw_parts)
            self.evidence_citations = self._evidence_filter.citations
            # Le texte affiché fait foi : c'est lui qui est persisté et contrôlé. Le rstrip
            # ne porte que sur l'état interne — les espaces de fin sont déjà partis au fil
            # de l'eau et les retirer du flux n'aurait aucun sens.
            self.final_text = "".join(emitted).rstrip()

            if self.output_check is not None:
                try:
                    verdict = self.output_check(
                        self.final_text,
                        self.evidence_text,
                        self.evidence_citations,
                        sorted(self.image_pages_seen),
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "[génération] contrôle d'ancrage en échec (%s) — réponse émise", exc
                    )
                    verdict = None
                if verdict is not None:
                    verdict["action"] = "passed" if verdict.get("ok") else "flagged"
                    self.verification = verdict
        finally:
            self.trace["duration_ms"] = int((time.perf_counter() - t0) * 1000)
            if self.verification is not None:
                self.trace["verification_action"] = self.verification.get("action")
