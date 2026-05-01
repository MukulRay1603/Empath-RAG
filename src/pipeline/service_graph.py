"""Resource registry loader and matcher for EmpathRAG Core support routing.

The filename remains `service_graph.py` for compatibility, but the paper/demo
concept is a resource registry of service objects, not a graph algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


DEFAULT_SERVICE_GRAPH = Path("data/curated/service_graph.jsonl")


@dataclass(frozen=True)
class ServiceNode:
    service_id: str
    resource_name: str
    description: str
    urgency_level: str
    safety_tiers: list[str]
    route_types: list[str]
    audience: list[str]
    issue_types: list[str]
    confidentiality_status: str
    hours: str
    contact_mode: list[str]
    contact: str
    location: str
    source_url: str
    source_authority: str
    last_verified: str
    usage_modes: list[str]
    do_not_use_for: list[str]
    notes: str

    @classmethod
    def from_dict(cls, row: dict) -> "ServiceNode":
        return cls(**row)

    def as_source(self, why: str = "resource registry match") -> dict:
        usage_mode = self.usage_modes[0] if self.usage_modes else "retrieval"
        return {
            "text": self.description,
            "title": self.resource_name,
            "source_name": self.resource_name,
            "url": self.source_url,
            "topic": ",".join(self.route_types),
            "risk_level": self.urgency_level,
            "usage_mode": usage_mode,
            "source_type": self.source_authority,
            "why_retrieved": why,
        }


def load_service_graph(path: Path | str = DEFAULT_SERVICE_GRAPH) -> list[ServiceNode]:
    graph_path = Path(path)
    if not graph_path.exists():
        return []
    nodes: list[ServiceNode] = []
    with graph_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            nodes.append(ServiceNode.from_dict(json.loads(line)))
    return nodes


def match_services(
    route: str,
    safety_tier: str,
    audience_mode: str = "student",
    limit: int = 5,
    path: Path | str = DEFAULT_SERVICE_GRAPH,
) -> list[ServiceNode]:
    nodes = load_service_graph(path)
    scored: list[tuple[int, ServiceNode]] = []
    for node in nodes:
        score = 0
        if route in node.route_types:
            score += 8
        if safety_tier in node.safety_tiers:
            score += 5
        if audience_mode in node.audience:
            score += 3
        if audience_mode == "helping_friend" and "friend_peer" in node.audience:
            score += 4
        if score > 0 and route not in node.do_not_use_for:
            scored.append((score, node))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [node for _, node in scored[:limit]]
