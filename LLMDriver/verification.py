"""
Verification-Enhanced Retrieval (KoMA-RAG Module 4).

Composite verification score (manuscript):
    V(e_j, Ω_i(t)) = V_semantic · V_factual · V_contextual

V_semantic   : cosine similarity of embeddings
V_factual    : deterministic schema/rule check by default;
               optional LLM judge when FACTUAL_USE_LLM=true
V_contextual : exp(-λ_time |t_j - t|) · I[scenario match]

Empty verified set: fallback to similarity-only Top-k when configured,
otherwise proceed with no few-shot memories (manuscript caveat).
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from rich import print
except ImportError:  # pragma: no cover
    pass

from LLMDriver.llm_backend import create_chat_llm


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    va = np.asarray(a, dtype=float)
    vb = np.asarray(b, dtype=float)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.clip(np.dot(va, vb) / (na * nb), 0.0, 1.0))


def _extract_lane_count(text: str) -> Optional[int]:
    m = re.search(r"road with\s+(\d+)\s+lanes?", text, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)\s+lanes?", text, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def _extract_scenario_tag(text: str, default: str = "highway") -> str:
    t = (text or "").lower()
    if "ramp" in t or "merge" in t:
        return "merge"
    if "roundabout" in t:
        return "roundabout"
    if "intersection" in t:
        return "intersection"
    if "highway" in t:
        return "highway"
    return default


@dataclass
class VerificationResult:
    experience: Dict[str, Any]
    score: float
    v_semantic: float
    v_factual: float
    v_contextual: float
    passed: bool


class VerificationModule:
    """Product-form composite verification for retrieved memory items."""

    def __init__(
        self,
        tau_verify: float = 0.5,
        lambda_time: float = 0.01,
        factual_use_llm: bool = False,
        empty_fallback: str = "similarity",  # "similarity" | "none"
        scenario_type: str = "highway",
        verbose: bool = False,
    ) -> None:
        self.tau_verify = tau_verify
        self.lambda_time = lambda_time
        self.factual_use_llm = factual_use_llm
        self.empty_fallback = empty_fallback
        self.scenario_type = scenario_type
        self.verbose = verbose
        self.llm = None
        if factual_use_llm:
            self.llm = create_chat_llm(role="verify", temperature=0.0, max_tokens=200)
        self._all_scores: List[float] = []
        self._factual_scores: List[float] = []
        self._passed = 0
        self._filtered = 0

    def semantic_score(
        self,
        query_embedding: Optional[Sequence[float]],
        doc_embedding: Optional[Sequence[float]],
        similarity_distance: Optional[float] = None,
    ) -> float:
        if query_embedding is not None and doc_embedding is not None:
            return _cosine(query_embedding, doc_embedding)
        # Chroma returns L2 distance by default for many embeddings; map to (0,1]
        if similarity_distance is not None:
            # smaller distance => higher similarity
            return float(1.0 / (1.0 + max(float(similarity_distance), 0.0)))
        return 0.5

    def factual_score_deterministic(self, experience_text: str, context: str) -> float:
        """Non-LLM factual check: lane count / merge-vs-highway agreement."""
        score = 1.0
        exp_lanes = _extract_lane_count(experience_text)
        ctx_lanes = _extract_lane_count(context)
        if exp_lanes is not None and ctx_lanes is not None:
            if exp_lanes == ctx_lanes:
                score *= 1.0
            elif abs(exp_lanes - ctx_lanes) == 1:
                score *= 0.6
            else:
                score *= 0.2

        exp_tag = _extract_scenario_tag(experience_text, self.scenario_type)
        ctx_tag = _extract_scenario_tag(context, self.scenario_type)
        if exp_tag != ctx_tag:
            score *= 0.3
        return float(np.clip(score, 0.0, 1.0))

    def factual_score_llm(self, experience_text: str, context: str) -> float:
        if self.llm is None:
            return self.factual_score_deterministic(experience_text, context)
        prompt = (
            "Is this past driving experience factually applicable to the current scenario?\n"
            f"Past: {experience_text[:400]}\n"
            f"Current: {context[:400]}\n"
            'Reply JSON only: {"is_consistent": true, "confidence": 0.8}'
        )
        resp = self.llm([
            {"role": "system", "content": "Reply with valid JSON only."},
            {"role": "user", "content": prompt},
        ])
        content = resp.content
        conf_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', content)
        conf = float(conf_match.group(1)) if conf_match else 0.7
        flag_match = re.search(r'"is_consistent"\s*:\s*(true|false)', content, flags=re.IGNORECASE)
        consistent = bool(flag_match and flag_match.group(1).lower() == "true")
        return float(np.clip(conf if consistent else 0.2, 0.0, 1.0))

    def factual_score(self, experience_text: str, context: str) -> float:
        if self.factual_use_llm:
            return self.factual_score_llm(experience_text, context)
        return self.factual_score_deterministic(experience_text, context)

    def contextual_score(
        self,
        experience_text: str,
        experience_time: Optional[float],
        current_time: float,
        context: str,
    ) -> float:
        t_j = float(experience_time) if experience_time is not None else current_time
        temporal = math.exp(-self.lambda_time * abs(t_j - current_time))
        exp_tag = _extract_scenario_tag(experience_text, self.scenario_type)
        ctx_tag = _extract_scenario_tag(context, self.scenario_type)
        scenario_match = 1.0 if exp_tag == ctx_tag else 0.0
        # Soften hard zero when tags unknown/default-equal
        if exp_tag == ctx_tag:
            scenario_match = 1.0
        else:
            scenario_match = 0.0
        return float(np.clip(temporal * (1.0 if scenario_match else 0.0), 0.0, 1.0))

    def verify_one(
        self,
        meta: Dict[str, Any],
        context: str,
        current_time: float,
        query_embedding: Optional[Sequence[float]] = None,
        doc_embedding: Optional[Sequence[float]] = None,
        similarity_distance: Optional[float] = None,
    ) -> VerificationResult:
        experience_text = (
            meta.get("sce_description")
            or meta.get("page_content")
            or meta.get("human_question")
            or ""
        )
        v_sem = self.semantic_score(query_embedding, doc_embedding, similarity_distance)
        v_fac = self.factual_score(experience_text, context)
        self._factual_scores.append(v_fac)
        v_ctx = self.contextual_score(
            experience_text,
            meta.get("simulation_time", meta.get("timestamp")),
            current_time,
            context,
        )
        # Product form (manuscript)
        composite = float(v_sem * v_fac * v_ctx)
        self._all_scores.append(composite)
        passed = composite > self.tau_verify
        return VerificationResult(
            experience=meta,
            score=composite,
            v_semantic=v_sem,
            v_factual=v_fac,
            v_contextual=v_ctx,
            passed=passed,
        )

    def filter_memories(
        self,
        candidates: List[Tuple[Dict[str, Any], Optional[float]]],
        context: str,
        current_time: float,
        top_k: int,
        query_embedding: Optional[Sequence[float]] = None,
        apply_filter: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        candidates: list of (metadata_dict, chroma_distance_or_None)
        Returns up to top_k verified (or similarity) memories.
        """
        if not candidates:
            return []

        scored: List[VerificationResult] = []
        for meta, dist in candidates:
            vr = self.verify_one(
                meta=meta,
                context=context,
                current_time=current_time,
                query_embedding=query_embedding,
                similarity_distance=dist,
            )
            scored.append(vr)

        if not apply_filter:
            scored.sort(key=lambda r: r.score, reverse=True)
            return [r.experience for r in scored[:top_k]]

        passed = [r for r in scored if r.passed]
        passed.sort(key=lambda r: r.score, reverse=True)
        selected = [r.experience for r in passed[:top_k]]
        self._passed += len(selected)
        self._filtered += max(0, len(scored) - len(selected))

        if not selected:
            if self.empty_fallback == "similarity":
                # Manuscript caveat / recommended fallback
                scored.sort(key=lambda r: r.v_semantic, reverse=True)
                selected = [r.experience for r in scored[:top_k]]
                if self.verbose:
                    print("[yellow]Verification emptied memory set; falling back to similarity Top-k[/yellow]")
            else:
                if self.verbose:
                    print("[yellow]Verification emptied memory set; proceeding without few-shot[/yellow]")
                selected = []

        if self.verbose and scored:
            mean_v = float(np.mean([r.score for r in scored]))
            print(f"[Verification] candidates={len(scored)} selected={len(selected)} mean_V={mean_v:.3f}")
        return selected

    def mean_score(self) -> float:
        return float(np.mean(self._all_scores)) if self._all_scores else 0.0

    def mean_factual_score(self) -> float:
        """Mean V_factual across verify_one() calls since the last reset (NaN if none)."""
        return float(np.mean(self._factual_scores)) if self._factual_scores else float("nan")

    def reset_episode_stats(self) -> None:
        self._all_scores = []
        self._factual_scores = []
        self._passed = 0
        self._filtered = 0
