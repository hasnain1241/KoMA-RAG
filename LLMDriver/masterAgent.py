"""
Master Coordination Module (KoMA-RAG Module 3).

Implements conflict detection, priority assignment, and directive broadcast
as described in Koma-Rag-Messam.tex. The Master does NOT solve a formal MDP;
it issues prompt/heuristic directives (goal, priority, constraints).

Dual-path / T_max / async Master are config stubs only (not evaluated).
"""

from __future__ import annotations

import json
import re
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from rich import print
except ImportError:  # pragma: no cover
    pass

from LLMDriver.llm_backend import create_chat_llm


@dataclass
class AgentKinematics:
    agent_id: str
    x: float
    y: float
    velocity: float
    lane: int
    on_ramp: bool = False
    is_idm: bool = False


@dataclass
class CoordinationDirective:
    agent_id: str
    assigned_goal: str
    priority: float
    constraints: List[str] = field(default_factory=list)
    has_conflict: bool = False
    timestamp: float = 0.0

    def to_prompt_block(self) -> str:
        cons = "; ".join(self.constraints) if self.constraints else "none"
        return (
            f"Master directive for agent {self.agent_id}:\n"
            f"- Assigned goal: {self.assigned_goal}\n"
            f"- Priority: {self.priority if self.priority != float('inf') else 'INF (yield to IDM)'}\n"
            f"- Constraints: {cons}\n"
            f"- Conflict active: {self.has_conflict}"
        )


class MasterAgent:
    """
    Hierarchical supervisor for LLM-controlled ego agents.

    Conflict detection projects goals into spatiotemporal target regions
    (manuscript Eq. for I_conflict). IDM vehicles receive infinite priority.
    """

    def __init__(
        self,
        delta_t_safe: float = 2.0,
        lambda_coop: float = 0.3,
        lambda_conflict: float = 0.5,
        conflict_only_llm: bool = True,
        use_llm: bool = True,
        verbose: bool = False,
    ) -> None:
        self.delta_t_safe = delta_t_safe
        self.lambda_coop = lambda_coop
        self.lambda_conflict = lambda_conflict
        self.conflict_only_llm = conflict_only_llm
        self.use_llm = use_llm
        self.verbose = verbose
        self.n_conflicts = 0
        self.n_resolved = 0
        self.llm = None
        if use_llm:
            self.llm = create_chat_llm(role="master", temperature=0.0, max_tokens=800)

    def _spatial_target(self, s: AgentKinematics) -> Tuple[float, float, int]:
        h = self.delta_t_safe
        return s.x, s.x + max(s.velocity, 0.0) * h, s.lane

    def detect_conflicts(
        self, states: Dict[str, AgentKinematics]
    ) -> List[Tuple[str, str]]:
        """Identify overlapping spatiotemporal targets within unsafe temporal margin."""
        aids = list(states.keys())
        conflicts: List[Tuple[str, str]] = []
        for i, a1 in enumerate(aids):
            for a2 in aids[i + 1:]:
                s1, s2 = states[a1], states[a2]
                x1s, x1e, l1 = self._spatial_target(s1)
                x2s, x2e, l2 = self._spatial_target(s2)
                spatial = (l1 == l2 and x1s < x2e and x2s < x1e)
                dv = abs(s1.velocity - s2.velocity)
                gap = abs(s1.x - s2.x)
                t_close = gap / dv if dv > 0.1 else (0.0 if gap < 15.0 else float("inf"))
                # Also flag close same-lane proximity even with similar speeds
                proximity = (l1 == l2 and gap < 20.0)
                if (spatial and t_close < self.delta_t_safe) or proximity:
                    conflicts.append((a1, a2))
                    self.n_conflicts += 1
        return conflicts

    def compute_priorities(
        self,
        ego_states: Dict[str, AgentKinematics],
        idm_states: Optional[Dict[str, AgentKinematics]] = None,
    ) -> Dict[str, float]:
        """IDM vehicles get priority=inf (conservative yield policy)."""
        priorities: Dict[str, float] = {}
        if idm_states:
            for aid in idm_states:
                priorities[aid] = float("inf")
        for aid, s in ego_states.items():
            base = min(1.0, 0.4 + 0.4 * (s.velocity / 30.0))
            if s.on_ramp:
                base = min(1.0, base + 0.3)
            priorities[aid] = base
        return priorities

    def _rule_based_directives(
        self,
        ego_states: Dict[str, AgentKinematics],
        proposed_goals: Dict[str, str],
        conflicts: List[Tuple[str, str]],
        priorities: Dict[str, float],
    ) -> Dict[str, CoordinationDirective]:
        conflict_set = {aid for pair in conflicts for aid in pair}
        msgs: Dict[str, CoordinationDirective] = {}
        for aid, s in ego_states.items():
            goal = proposed_goals.get(aid, "Navigate safely and avoid collisions")
            constraints: List[str] = []
            has_conflict = aid in conflict_set
            if has_conflict:
                other_ids = [b for a, b in conflicts if a == aid] + [
                    a for a, b in conflicts if b == aid
                ]
                for oid in other_ids:
                    op = priorities.get(oid, 0.5)
                    my_p = priorities.get(aid, 0.5)
                    if op == float("inf") or op > my_p:
                        constraints.append(f"Yield to agent {oid}")
                        if s.on_ramp:
                            goal = "Slow down and wait for a safe merge gap"
                        else:
                            goal = "Maintain safe distance; prefer IDLE/decelerate over aggressive lane change"
                    else:
                        constraints.append(f"Proceed with priority over agent {oid}")
                self.n_resolved += 1
            msgs[aid] = CoordinationDirective(
                agent_id=aid,
                assigned_goal=goal,
                priority=float(priorities.get(aid, 0.5)),
                constraints=constraints,
                has_conflict=has_conflict,
                timestamp=time.time(),
            )
        return msgs

    def _llm_refine_directives(
        self,
        ego_states: Dict[str, AgentKinematics],
        proposed_goals: Dict[str, str],
        conflicts: List[Tuple[str, str]],
        base_msgs: Dict[str, CoordinationDirective],
    ) -> Dict[str, CoordinationDirective]:
        """Optional LLM enrichment when conflicts exist (prompt-based Master)."""
        if self.llm is None:
            return base_msgs

        state_lines = []
        for aid, s in ego_states.items():
            state_lines.append(
                f"- Agent {aid}: x={s.x:.1f}, y={s.y:.1f}, v={s.velocity:.1f} m/s, "
                f"lane={s.lane}, on_ramp={s.on_ramp}, proposed_goal={proposed_goals.get(aid, 'n/a')}"
            )
        conflict_txt = ", ".join(f"({a},{b})" for a, b in conflicts) or "none"
        base_txt = "\n".join(m.to_prompt_block() for m in base_msgs.values())

        system = textwrap.dedent("""\
            You are the Master Coordination Agent for multi-agent highway driving.
            Resolve conflicts with discrete directives. Reply with JSON only:
            {
              "directives": [
                {"agent_id": "0", "assigned_goal": "...", "priority": 0.7,
                 "constraints": ["Yield to agent 1"]}
              ]
            }
            Priorities in [0,1]. Prefer yielding to higher-priority / IDM traffic.
            """)
        human = (
            f"Global ego states:\n" + "\n".join(state_lines) +
            f"\nConflicts: {conflict_txt}\n"
            f"Rule-based draft directives:\n{base_txt}\n"
            "Return refined directives JSON."
        )
        try:
            resp = self.llm([
                {"role": "system", "content": system},
                {"role": "user", "content": human},
            ])
            content = resp.content
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if not match:
                raise ValueError("No JSON object in Master response")
            data = json.loads(match.group(0))
            for item in data.get("directives", []):
                aid = str(item.get("agent_id"))
                if aid not in base_msgs:
                    continue
                base_msgs[aid].assigned_goal = str(item.get("assigned_goal", base_msgs[aid].assigned_goal))
                try:
                    base_msgs[aid].priority = float(item.get("priority", base_msgs[aid].priority))
                except (TypeError, ValueError):
                    pass
                cons = item.get("constraints")
                if isinstance(cons, list):
                    base_msgs[aid].constraints = [str(c) for c in cons]
            if self.verbose:
                print("[MasterAgent] LLM-refined directives applied")
        except Exception as exc:
            # Fail clearly for API errors; keep rule-based if JSON parse fails only
            msg = str(exc).lower()
            if "api error" in msg or "request failed" in msg or "api key" in msg:
                raise
            print(f"[yellow]Master LLM parse fallback to rule-based directives: {exc}[/yellow]")
        return base_msgs

    def coordinate(
        self,
        ego_states: Dict[str, AgentKinematics],
        proposed_goals: Dict[str, str],
        idm_states: Optional[Dict[str, AgentKinematics]] = None,
        enable: bool = True,
    ) -> Dict[str, CoordinationDirective]:
        """
        Broadcast msg_M→i = {g_assigned, π_assigned, constraints}.

        If enable=False, returns pass-through defaults (Base KoMA ablation).
        """
        if not enable:
            return {
                aid: CoordinationDirective(
                    agent_id=aid,
                    assigned_goal=proposed_goals.get(aid, "Navigate safely and avoid collisions"),
                    priority=0.5,
                    constraints=[],
                    has_conflict=False,
                    timestamp=time.time(),
                )
                for aid in ego_states
            }

        priorities = self.compute_priorities(ego_states, idm_states)
        conflicts = self.detect_conflicts(ego_states)
        msgs = self._rule_based_directives(ego_states, proposed_goals, conflicts, priorities)

        invoke_llm = self.use_llm and (
            (not self.conflict_only_llm) or bool(conflicts)
        )
        if invoke_llm:
            msgs = self._llm_refine_directives(ego_states, proposed_goals, conflicts, msgs)

        if self.verbose:
            print(f"[MasterAgent] conflicts={len(conflicts)} resolved={self.n_resolved}")
            for m in msgs.values():
                print(m.to_prompt_block())
        return msgs

    @staticmethod
    def extract_states_from_env(env, sce) -> Tuple[Dict[str, AgentKinematics], Dict[str, AgentKinematics]]:
        """Build ego/IDM kinematics from highway-env + EnvScenario."""
        ego_states: Dict[str, AgentKinematics] = {}
        idm_states: Dict[str, AgentKinematics] = {}
        controlled = list(getattr(env, "controlled_vehicles", []) or [])
        for i, veh in enumerate(controlled):
            lane = 0
            try:
                lane = int(veh.lane_index[2]) if veh.lane_index is not None else 0
            except Exception:
                lane = 0
            on_ramp = False
            try:
                on_ramp = bool(veh.lane_index and veh.lane_index[0] in ("k", "b") and
                               (veh.lane_index[0] == "k" or (
                                   veh.lane_index[0] == "b" and
                                   len(sce.network.all_side_lanes(veh.lane_index)) == 1
                               )))
            except Exception:
                on_ramp = False
            ego_states[str(i)] = AgentKinematics(
                agent_id=str(i),
                x=float(veh.position[0]),
                y=float(veh.position[1]),
                velocity=float(veh.speed),
                lane=lane,
                on_ramp=on_ramp,
                is_idm=False,
            )

        for j, veh in enumerate(getattr(env.road, "vehicles", []) or []):
            if veh in controlled:
                continue
            lane = 0
            try:
                lane = int(veh.lane_index[2]) if veh.lane_index is not None else 0
            except Exception:
                lane = 0
            idm_states[f"idm_{j}"] = AgentKinematics(
                agent_id=f"idm_{j}",
                x=float(veh.position[0]),
                y=float(veh.position[1]),
                velocity=float(veh.speed),
                lane=lane,
                on_ramp=False,
                is_idm=True,
            )
        return ego_states, idm_states
