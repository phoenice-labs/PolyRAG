"""
Phase 11: LLM-Enhanced Entity/Relation Extractor.

Uses LM Studio (any OpenAI-compatible endpoint) to extract rich entity-relation
triples from text chunks via structured JSON prompting.

Complements the spaCy extractor (Phase 10):
  - spaCy: fast, offline, reliable NER
  - LLM:   richer relations, domain-aware, understands implicit relationships
  - Combined (merge_with_spacy=True): highest quality — both run, results merged

Config (config.yaml):
  graph:
    llm_extraction:
      enabled: true
      merge_with_spacy: true   # false = LLM only (no spaCy)
      max_chunk_chars: 2000
"""
from __future__ import annotations

import json
import logging
import re
from typing import List

from core.graph.models import Entity, ExtractionResult, Triple, SPACY_TYPE_MAP, make_entity_id

logger = logging.getLogger(__name__)

_EXTRACTION_PROMPT = """\
Extract all named entities and their relationships from the text below.
Return ONLY a JSON object — no explanation, no markdown:

{{
  "entities": [
    {{"text": "Entity Name", "type": "PERSON|ORG|LOCATION|DATE|EVENT|LAW|PRODUCT|CONCEPT"}}
  ],
  "relations": [
    {{"subject": "Entity A", "predicate": "relationship_verb", "object": "Entity B"}}
  ]
}}

Rules:
- Use short lowercase verbs for predicates: "founded", "governs", "works_for", "references", "co_occurs"
- Only include relations between entities you listed
- If no entities, return {{"entities": [], "relations": []}}

Text:
{text}
"""

_VALID_TYPES = set(SPACY_TYPE_MAP.values()) | {"CONCEPT"}


class LLMEntityExtractor:
    """
    LLM-based entity and relation extractor using LM Studio.

    Falls back gracefully when LM Studio is offline (returns empty ExtractionResult).
    """

    def __init__(
        self,
        llm_client,
        max_chunk_chars: int = 2000,
        confidence: float = 0.75,
    ) -> None:
        self._client = llm_client
        self.max_chunk_chars = max_chunk_chars
        self.confidence = confidence

    def is_available(self) -> bool:
        try:
            return self._client.is_available()
        except Exception:
            return False

    def extract(self, text: str, chunk_id: str) -> ExtractionResult:
        """Extract entities and relations from text via LLM JSON prompting."""
        if not text or not text.strip():
            return ExtractionResult(chunk_id=chunk_id)

        # Load prompt from registry (with fallback to hardcoded _EXTRACTION_PROMPT)
        try:
            from core.prompt_registry import PromptRegistry
            prompt_template = PromptRegistry.instance().get_prompt("graph_entity_extractor")
        except Exception:
            prompt_template = _EXTRACTION_PROMPT

        # Use simple replace instead of .format() — the template may contain
        # JSON braces like {"entities": ...} that .format() mis-interprets as placeholders.
        prompt = prompt_template.replace("{text}", text[:self.max_chunk_chars])
        try:
            raw = self._client.complete(
                prompt,
                system="You are a precise information extraction assistant. Return only valid JSON.",
                max_tokens=1024,
                temperature=0.0,
                trace_method="LLM Graph Entity Extraction",
            )
            return self._parse(raw, chunk_id)
        except Exception as exc:
            logger.debug("LLM extraction failed for chunk %s: %s", chunk_id, exc)
            return ExtractionResult(chunk_id=chunk_id)

    def _parse(self, raw: str, chunk_id: str) -> ExtractionResult:
        """Parse LLM JSON response into ExtractionResult."""
        # Strip markdown fences
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            logger.debug("No JSON block in LLM response: %.100s", raw)
            return ExtractionResult(chunk_id=chunk_id)

        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return ExtractionResult(chunk_id=chunk_id)

        entities: List[Entity] = []
        entity_map: dict = {}   # text.lower() → Entity
        seen: set = set()

        for e in data.get("entities", []):
            text_val = str(e.get("text", "")).strip()
            type_val = str(e.get("type", "CONCEPT")).upper()
            if len(text_val) < 2:
                continue
            if type_val not in _VALID_TYPES:
                type_val = "CONCEPT"
            eid = make_entity_id(type_val, text_val)
            if eid not in seen:
                seen.add(eid)
                ent = Entity(entity_id=eid, text=text_val, entity_type=type_val)
                entities.append(ent)
                entity_map[text_val.lower()] = ent

        triples: List[Triple] = []
        for r in data.get("relations", []):
            subj_t = str(r.get("subject", "")).strip().lower()
            pred   = str(r.get("predicate", "")).strip().lower()
            obj_t  = str(r.get("object", "")).strip().lower()
            if not (subj_t and pred and obj_t):
                continue
            subj_ent = entity_map.get(subj_t)
            obj_ent  = entity_map.get(obj_t)
            if subj_ent and obj_ent and pred:
                triples.append(Triple(
                    subject=subj_ent, predicate=pred, object=obj_ent,
                    confidence=self.confidence, chunk_id=chunk_id,
                ))

        return ExtractionResult(chunk_id=chunk_id, entities=entities, triples=triples)
