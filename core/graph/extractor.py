"""
EntityRelationExtractor — NER + dependency-parse relation extraction.

Uses spaCy (en_core_web_sm, ~12 MB) for:
  1. Named Entity Recognition:  PERSON, ORG, LOCATION, DATE, EVENT, LAW, PRODUCT, CONCEPT
  2. Noun chunk indexing: root-lemma CONCEPT entities for all meaningful noun phrases
  3. Relation extraction via Subject-Verb-Object dependency parsing
  4. Co-occurrence relations: any two entities in the same sentence

Falls back gracefully if spaCy is not installed (returns empty ExtractionResult).

Install:
  pip install spacy
  python -m spacy download en_core_web_sm
"""
from __future__ import annotations

import re
from typing import List, Optional

from core.graph.models import Entity, ExtractionResult, Relation, Triple, make_entity_id, SPACY_TYPE_MAP

# Common stopword-only noun chunks to skip even if they pass is_stop check
_SKIP_LEMMAS = frozenset({"it", "he", "she", "they", "we", "i", "you", "that", "this", "what"})


class EntityRelationExtractor:
    """
    Extracts entities and relations from text chunks.

    Parameters
    ----------
    model : spaCy model name (default: en_core_web_sm)
    min_entity_len : minimum characters for an entity surface form
    extract_svo   : whether to extract subject-verb-object triples
    extract_cooc  : whether to extract sentence-level co-occurrence edges
    """

    def __init__(
        self,
        model: str = "en_core_web_sm",
        min_entity_len: int = 2,
        extract_svo: bool = True,
        extract_cooc: bool = True,
        extract_noun_chunks: bool = True,
    ) -> None:
        self.model_name = model
        self.min_entity_len = min_entity_len
        self.extract_svo = extract_svo
        self.extract_cooc = extract_cooc
        self.extract_noun_chunks = extract_noun_chunks
        self._nlp = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Return True if spaCy and the model are installed."""
        try:
            import spacy
            spacy.load(self.model_name)
            return True
        except Exception:
            return False

    def _load(self):
        if self._nlp is not None:
            return
        try:
            import spacy
            self._nlp = spacy.load(self.model_name)
        except OSError:
            raise RuntimeError(
                f"spaCy model '{self.model_name}' not found.\n"
                f"Install it with:  python -m spacy download {self.model_name}"
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(self, text: str, chunk_id: str) -> ExtractionResult:
        """Extract entities and relations from a text chunk.

        Returns an empty ExtractionResult if spaCy is unavailable.
        """
        if not text or not text.strip():
            return ExtractionResult(chunk_id=chunk_id)

        try:
            self._load()
        except Exception:
            return ExtractionResult(chunk_id=chunk_id)

        doc = self._nlp(text[:50_000])  # guard against very large chunks
        entities = self._extract_entities(doc, chunk_id)
        triples = []

        if self.extract_svo:
            triples += self._extract_svo(doc, chunk_id)
        if self.extract_cooc:
            triples += self._extract_cooccurrence(doc, chunk_id, entities)

        return ExtractionResult(
            chunk_id=chunk_id,
            entities=entities,
            triples=triples,
        )

    def extract_entities_only(self, text: str) -> List[Entity]:
        """Lightweight entity detection for query-time use (no relation extraction)."""
        if not text:
            return []
        try:
            self._load()
        except Exception:
            return []
        doc = self._nlp(text[:2000])
        return self._extract_entities(doc, chunk_id="query")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_entities(self, doc, chunk_id: str) -> List[Entity]:
        seen: set = set()
        entities: List[Entity] = []

        # 1. Named Entity Recognition — typed entities (PERSON, ORG, DATE, …)
        for ent in doc.ents:
            if len(ent.text.strip()) < self.min_entity_len:
                continue
            if ent.label_ not in SPACY_TYPE_MAP:
                continue
            entity = Entity.from_spacy(ent.text.strip(), ent.label_)
            if entity.entity_id not in seen:
                seen.add(entity.entity_id)
                entities.append(entity)

        # 2. Noun chunk CONCEPT indexing — catches verbs-turned-nouns, common nouns,
        #    and noun phrases that spaCy NER misses (e.g. "surrender", "lands", "import").
        #    We use the root token's lemma as the canonical form so "the surrender of those
        #    lands" and "his surrender" both collapse to CONCEPT:surrender.
        if self.extract_noun_chunks:
            for nc in doc.noun_chunks:
                root = nc.root
                # Skip if root is a stop word, very short, or a pronoun
                lemma = root.lemma_.lower().strip()
                # For proper nouns, use surface text — spaCy lemmatizes them poorly
                canonical = root.text.lower().strip() if root.pos_ == "PROPN" else lemma
                if (
                    root.is_stop
                    or root.is_punct
                    or root.pos_ == "PRON"
                    or len(canonical) < max(self.min_entity_len, 3)
                    or canonical in _SKIP_LEMMAS
                ):
                    continue
                # Skip if a NER entity already covers this token (avoid duplicates)
                if root.ent_type_ in SPACY_TYPE_MAP:
                    continue
                entity = Entity(
                    entity_id=make_entity_id("CONCEPT", canonical),
                    text=canonical,
                    entity_type="CONCEPT",
                    aliases=[nc.text.strip()] if nc.text.strip().lower() != canonical else [],
                )
                if entity.entity_id not in seen:
                    seen.add(entity.entity_id)
                    entities.append(entity)

        return entities

    def _extract_svo(self, doc, chunk_id: str) -> List[Triple]:
        """Extract Subject → Verb → Object triples via dependency parsing."""
        triples: List[Triple] = []
        for sent in doc.sents:
            for token in sent:
                # Find verbs with nominal subjects
                if token.pos_ not in ("VERB", "AUX"):
                    continue
                subjects = [
                    child for child in token.children
                    if child.dep_ in ("nsubj", "nsubjpass") and child.ent_type_
                ]
                objects = [
                    child for child in token.children
                    if child.dep_ in ("dobj", "pobj", "attr", "nsubjpass") and child.ent_type_
                ]
                for subj in subjects:
                    for obj in objects:
                        if subj.ent_type_ not in SPACY_TYPE_MAP:
                            continue
                        if obj.ent_type_ not in SPACY_TYPE_MAP:
                            continue
                        verb_lemma = token.lemma_.lower()
                        if not verb_lemma or len(verb_lemma) < 2:
                            continue
                        subject_ent = Entity.from_spacy(subj.text, subj.ent_type_)
                        object_ent = Entity.from_spacy(obj.text, obj.ent_type_)
                        triples.append(Triple(
                            subject=subject_ent,
                            predicate=verb_lemma,
                            object=object_ent,
                            confidence=0.8,
                            chunk_id=chunk_id,
                        ))
        return triples

    def _extract_cooccurrence(
        self, doc, chunk_id: str, entities: List[Entity]
    ) -> List[Triple]:
        """Add co-occurrence edges between any two entities in the same sentence."""
        triples: List[Triple] = []
        seen_pairs: set = set()

        # Map char offsets to entities for fast lookup
        entity_map = {e.text: e for e in entities}

        for sent in doc.sents:
            sent_ents = [
                Entity.from_spacy(ent.text.strip(), ent.label_)
                for ent in sent.ents
                if len(ent.text.strip()) >= self.min_entity_len
                and ent.label_ in SPACY_TYPE_MAP
            ]
            for i, ent_a in enumerate(sent_ents):
                for ent_b in sent_ents[i + 1:]:
                    pair = tuple(sorted([ent_a.entity_id, ent_b.entity_id]))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    triples.append(Triple(
                        subject=ent_a,
                        predicate="co_occurs",
                        object=ent_b,
                        confidence=0.5,
                        chunk_id=chunk_id,
                    ))
        return triples
