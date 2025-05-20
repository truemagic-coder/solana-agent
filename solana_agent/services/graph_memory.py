import datetime
import uuid
from typing import Dict, Any, List
from solana_agent.adapters.openai_adapter import OpenAIAdapter
from solana_agent.adapters.pinecone_adapter import PineconeAdapter
from solana_agent.adapters.mongodb_graph_adapter import MongoDBGraphAdapter
from solana_agent.interfaces.services.graph_memory import (
    GraphMemoryService as GraphMemoryServiceInterface,
)


class GraphMemoryService(GraphMemoryServiceInterface):
    def __init__(
        self,
        graph_adapter: MongoDBGraphAdapter,
        pinecone_adapter: PineconeAdapter,
        openai_adapter: OpenAIAdapter,
        embedding_model: str = "text-embedding-3-large",
    ):
        self.graph = graph_adapter
        self.pinecone = pinecone_adapter
        self.openai = openai_adapter
        self.embedding_model = embedding_model

    async def add_episode(
        self,
        user_message: str,
        assistant_message: str,
        user_id: str,
    ):
        entities = [
            {"type": "user_message", "text": user_message, "user_id": user_id},
            {
                "type": "assistant_message",
                "text": assistant_message,
                "user_id": user_id,
            },
        ]
        episode_id = str(uuid.uuid4())
        node_ids = []
        for entity in entities:
            entity["episode_id"] = episode_id
            node_id = await self.graph.add_node(entity)
            node_ids.append(node_id)
        edge = {
            "source": node_ids[0],
            "target": node_ids[1],
            "type": "reply",
            "episode_id": episode_id,
            "user_id": user_id,
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
        }
        await self.graph.add_edge(edge)
        # Save vectors in user-specific namespace
        namespace = f"{user_id}_memory"
        for node_id, entity in zip(node_ids, entities):
            embedding = await self.openai.embed_text(
                entity["text"], model=self.embedding_model
            )
            await self.pinecone.upsert(
                [{"id": node_id, "values": embedding}],
                namespace=namespace,
            )
        return episode_id

    async def search(
        self, query: str, user_id: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        embedding = await self.openai.embed_text(query, model=self.embedding_model)
        namespace = f"{user_id}_memory"
        results = await self.pinecone.query_and_rerank(
            vector=embedding,
            query_text_for_rerank=query,
            top_k=top_k,
            namespace=namespace,
        )
        node_ids = [r["id"] for r in results]
        # Only return nodes that match user_id
        nodes = []
        for nid in node_ids:
            node = await self.graph.get_node(nid)
            if node and node.get("user_id") == user_id:
                nodes.append(node)
        return nodes

    async def traverse(self, node_id: str, depth: int = 1) -> List[Dict[str, Any]]:
        return await self.graph.find_neighbors(node_id, depth=depth)
