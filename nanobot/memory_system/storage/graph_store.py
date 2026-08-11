"""GraphStore interface, in-memory backend, and Neo4j adapter."""

# Neo4j is an optional lazy import and exposes dynamic graph value objects.
# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any

from nanobot.memory_system.models import Entity, Relation, SubGraph

_NODE_TYPES = {
    "User",
    "Person",
    "Project",
    "Technology",
    "Preference",
    "Domain",
    "Concept",
    "Goal",
    "Event",
}
_RELATION_TYPES = {
    "INTERESTED_IN",
    "WORKS_ON",
    "USES",
    "PREFERS",
    "RELATES_TO",
    "MENTIONED_IN",
    "CONTRADICTS",
}


class GraphStore(ABC):
    @abstractmethod
    async def upsert_entity(self, entity: Entity, *, read_version: int | None = None) -> None: ...

    @abstractmethod
    async def upsert_relation(self, relation: Relation) -> None: ...

    @abstractmethod
    async def multi_hop_query(
        self,
        entity_ids: list[str],
        max_depth: int,
    ) -> list[SubGraph]: ...


class InMemoryGraphStore(GraphStore):
    def __init__(self) -> None:
        self.entities: dict[str, Entity] = {}
        self.relations: dict[str, Relation] = {}
        self._lock = asyncio.Lock()

    async def upsert_entity(self, entity: Entity, *, read_version: int | None = None) -> None:
        async with self._lock:
            current = self.entities.get(entity.id)
            if current is not None and read_version is not None and current.version != read_version:
                from nanobot.memory_system.storage.metadata_store import OptimisticLockError

                raise OptimisticLockError(entity.id, read_version, current.version)
            next_version = 1 if current is None else current.version + 1
            self.entities[entity.id] = entity.model_copy(update={"version": next_version})

    async def upsert_relation(self, relation: Relation) -> None:
        async with self._lock:
            self.relations[relation.id] = relation.model_copy(deep=True)

    async def multi_hop_query(
        self,
        entity_ids: list[str],
        max_depth: int,
    ) -> list[SubGraph]:
        async with self._lock:
            frontier = set(entity_ids)
            visited = set(frontier)
            relations: list[Relation] = []
            for _ in range(max_depth):
                next_frontier: set[str] = set()
                for relation in self.relations.values():
                    if relation.source_entity_id in frontier or relation.target_entity_id in frontier:
                        relations.append(relation.model_copy(deep=True))
                        next_frontier.update(
                            [relation.source_entity_id, relation.target_entity_id]
                        )
                frontier = next_frontier - visited
                visited.update(next_frontier)
                if not frontier:
                    break
            entities = [
                self.entities[entity_id].model_copy(deep=True)
                for entity_id in visited
                if entity_id in self.entities
            ]
        return [SubGraph(entities=entities, relations=relations)] if entities else []


class Neo4jGraphStore(GraphStore):
    def __init__(
        self,
        *,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
        initialize_schema: bool = True,
        driver: Any | None = None,
    ) -> None:
        try:
            from neo4j import AsyncGraphDatabase
        except ImportError as exc:
            raise RuntimeError(
                "Neo4j memory requires the 'memory' extra: pip install nanobot-ai[memory]"
            ) from exc
        self._driver = driver or AsyncGraphDatabase.driver(
            uri,
            auth=(username, password),
        )
        self._database = database
        self._initialize_schema = initialize_schema
        self._ready = False
        self._ready_lock = asyncio.Lock()

    async def _ensure_schema(self) -> None:
        if self._ready or not self._initialize_schema:
            return
        async with self._ready_lock:
            if self._ready:
                return
            statements = [
                f"CREATE CONSTRAINT {label.lower()}_memory_id IF NOT EXISTS "
                f"FOR (n:{label}) REQUIRE n.id IS UNIQUE"
                for label in sorted(_NODE_TYPES)
            ]
            async with self._driver.session(database=self._database) as session:
                for statement in statements:
                    await session.run(statement)
            self._ready = True

    async def upsert_entity(self, entity: Entity, *, read_version: int | None = None) -> None:
        from nanobot.memory_system.storage.metadata_store import OptimisticLockError

        await self._ensure_schema()
        label = entity.entity_type if entity.entity_type in _NODE_TYPES else "Concept"
        query = (
            f"MERGE (e:{label} {{id: $id}}) "
            "ON CREATE SET e.name=$name, e.attributes_json=$attributes_json, e.version=1 "
            "WITH e, e.version AS current_version "
            "WHERE $read_version IS NULL OR current_version = $read_version "
            "SET e.name=$name, e.attributes_json=$attributes_json, "
            "e.version=CASE WHEN $read_version IS NULL THEN current_version "
            "ELSE current_version + 1 END "
            "RETURN e.version AS version"
        )
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                query,
                id=entity.id,
                name=entity.name,
                attributes_json=json.dumps(entity.attributes, ensure_ascii=False),
                read_version=read_version,
            )
            row = await result.single()
            if row is None:
                raise OptimisticLockError(entity.id, read_version or 0, -1)

    async def upsert_relation(self, relation: Relation) -> None:
        await self._ensure_schema()
        rel_type = (
            relation.relation_type
            if relation.relation_type in _RELATION_TYPES
            else "RELATES_TO"
        )
        query = (
            "MATCH (s {id: $source}), (t {id: $target}) "
            f"MERGE (s)-[r:{rel_type} {{id: $id}}]->(t) "
            "SET r.weight=$weight, r.source_memory_id=$source_memory_id"
        )
        async with self._driver.session(database=self._database) as session:
            await session.run(
                query,
                id=relation.id,
                source=relation.source_entity_id,
                target=relation.target_entity_id,
                weight=relation.weight,
                source_memory_id=relation.source_memory_id,
            )

    async def multi_hop_query(
        self,
        entity_ids: list[str],
        max_depth: int,
    ) -> list[SubGraph]:
        if not entity_ids:
            return []
        await self._ensure_schema()
        depth = min(max(max_depth, 1), 4)
        query = (
            "MATCH p=(start)-[rels*1.."
            f"{depth}"
            "]-(neighbor) WHERE start.id IN $entity_ids "
            "RETURN nodes(p) AS nodes, relationships(p) AS relationships LIMIT 100"
        )
        entities: dict[str, Entity] = {}
        relations: dict[str, Relation] = {}
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, entity_ids=entity_ids)
            async for row in result:
                for node in row["nodes"]:
                    data = dict(node)
                    entity_id = str(data.get("id", ""))
                    if entity_id:
                        attributes_value = data.get("attributes_json")
                        if isinstance(attributes_value, str):
                            try:
                                parsed_attributes = json.loads(attributes_value)
                            except ValueError:
                                parsed_attributes = {}
                        else:
                            parsed_attributes = data.get("attributes") or {}
                        entities[entity_id] = Entity(
                            id=entity_id,
                            name=str(data.get("name", entity_id)),
                            entity_type=next(iter(node.labels), "Concept"),
                            attributes=(
                                parsed_attributes
                                if isinstance(parsed_attributes, dict)
                                else {}
                            ),
                            version=int(data.get("version", 1)),
                        )
                for edge in row["relationships"]:
                    data = dict(edge)
                    edge_id = str(data.get("id", edge.element_id))
                    relations[edge_id] = Relation(
                        id=edge_id,
                        source_entity_id=str(edge.start_node.get("id")),
                        target_entity_id=str(edge.end_node.get("id")),
                        relation_type=edge.type,
                        weight=float(data.get("weight", 1.0)),
                        source_memory_id=str(data.get("source_memory_id", "")),
                    )
        if not entities:
            return []
        return [SubGraph(entities=list(entities.values()), relations=list(relations.values()))]
