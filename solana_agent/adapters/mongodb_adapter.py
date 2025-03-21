"""
MongoDB adapter for the Solana Agent system.

This adapter implements the DataStorageProvider interface for MongoDB.
"""
import uuid
from typing import Dict, List, Tuple, Optional

from pymongo import MongoClient

from solana_agent.interfaces.providers.data_storage import DataStorageProvider


class MongoDBAdapter(DataStorageProvider):
    """MongoDB implementation of DataStorageProvider."""

    def __init__(self, connection_string: str, database_name: str):
        self.client = MongoClient(connection_string)
        self.db = self.client[database_name]

    def create_collection(self, name: str) -> None:
        if name not in self.db.list_collection_names():
            self.db.create_collection(name)

    def collection_exists(self, name: str) -> bool:
        return name in self.db.list_collection_names()

    def insert_one(self, collection: str, document: Dict) -> str:
        if "_id" not in document:
            document["_id"] = str(uuid.uuid4())
        self.db[collection].insert_one(document)
        return document["_id"]

    def find_one(self, collection: str, query: Dict) -> Optional[Dict]:
        return self.db[collection].find_one(query)

    def find(
        self,
        collection: str,
        query: Dict,
        sort: Optional[List[Tuple]] = None,
        limit: int = 0,
        skip: int = 0
    ) -> List[Dict]:
        cursor = self.db[collection].find(query)
        if sort:
            cursor = cursor.sort(sort)
        if limit > 0:
            cursor = cursor.limit(limit)
        if skip > 0:
            cursor = cursor.skip(skip)
        return list(cursor)

    def update_one(self, collection: str, query: Dict, update: Dict, upsert: bool = False) -> bool:
        result = self.db[collection].update_one(query, update, upsert=upsert)
        return result.modified_count > 0 or (upsert and result.upserted_id is not None)

    def delete_one(self, collection: str, query: Dict) -> bool:
        result = self.db[collection].delete_one(query)
        return result.deleted_count == 1

    def delete_all(self, collection: str, query: Dict) -> bool:
        total_documents = self.db[collection].count_documents(query)
        deleted_result = self.db[collection].delete_many(query)
        return deleted_result.deleted_count == total_documents

    def count_documents(self, collection: str, query: Dict) -> int:
        return self.db[collection].count_documents(query)

    def aggregate(self, collection: str, pipeline: List[Dict]) -> List[Dict]:
        return list(self.db[collection].aggregate(pipeline))

    def create_index(self, collection: str, keys: List[Tuple], **kwargs) -> None:
        self.db[collection].create_index(keys, **kwargs)
