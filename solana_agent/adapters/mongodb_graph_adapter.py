import uuid
from typing import Dict, Any, List, Optional
from solana_agent.interfaces.providers.graph_storage import GraphStorageProvider
from solana_agent.adapters.mongodb_adapter import MongoDBAdapter


class MongoDBGraphAdapter(GraphStorageProvider):
    def __init__(
        self,
        mongo_adapter: MongoDBAdapter,
        node_collection: str = "graph_nodes",
        edge_collection: str = "graph_edges",
    ):
        self.mongo = mongo_adapter
        self.node_collection = node_collection
        self.edge_collection = edge_collection

    async def add_node(self, node: Dict[str, Any]) -> str:
        node = dict(node)
        node["uuid"] = node.get("uuid", str(uuid.uuid4()))
        self.mongo.insert_one(self.node_collection, node)
        return node["uuid"]

    async def add_edge(self, edge: Dict[str, Any]) -> str:
        edge = dict(edge)
        edge["uuid"] = edge.get("uuid", str(uuid.uuid4()))
        return self.mongo.insert_one(self.edge_collection, edge)

    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        return self.mongo.find_one(self.node_collection, {"uuid": node_id})

    async def get_edges(
        self, node_id: str, direction: str = "both"
    ) -> List[Dict[str, Any]]:
        if direction == "out":
            query = {"source": node_id}
        elif direction == "in":
            query = {"target": node_id}
        else:
            query = {"$or": [{"source": node_id}, {"target": node_id}]}
        return self.mongo.find(self.edge_collection, query)

    async def find_neighbors(
        self, node_id: str, depth: int = 1
    ) -> List[Dict[str, Any]]:
        neighbors = set()
        current = {node_id}
        for _ in range(depth):
            edges = await self.get_edges(list(current)[0])
            for edge in edges:
                neighbors.add(edge.get("source"))
                neighbors.add(edge.get("target"))
            current = neighbors
        neighbors.discard(node_id)
        return [await self.get_node(nid) for nid in neighbors if nid]

    async def temporal_query(
        self, node_id: str, start_time: Optional[str], end_time: Optional[str]
    ) -> List[Dict[str, Any]]:
        query = {"$or": [{"source": node_id}, {"target": node_id}]}
        if start_time or end_time:
            query["timestamp"] = {}
            if start_time:
                query["timestamp"]["$gte"] = start_time
            if end_time:
                query["timestamp"]["$lte"] = end_time
        return self.mongo.find(self.edge_collection, query)
