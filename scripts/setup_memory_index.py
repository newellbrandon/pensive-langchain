#!/usr/bin/env python3
"""Create the vector search index for agent long-term memory."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pymongo import MongoClient
from pymongo.errors import OperationFailure

from config import Config

INDEX_NAME = "vector_index"
COLLECTION = "agent_memories"


def main() -> int:
    Config.validate()

    index_definition = {
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": Config.EMBEDDING_DIMENSIONS,
                "similarity": "cosine",
            },
            {"type": "filter", "path": "user_id"},
        ]
    }

    print(f"Database: {Config.MONGODB_DB}")
    print(f"Collection: {COLLECTION}")
    print(f"Index name: {INDEX_NAME}")
    print("Index definition:")
    print(json.dumps(index_definition, indent=2))

    client = MongoClient(Config.MONGODB_URI)
    db = client[Config.MONGODB_DB]
    collection = db[COLLECTION]

    try:
        existing = list(collection.list_search_indexes())
        for idx in existing:
            if idx.get("name") == INDEX_NAME:
                print(f"Index '{INDEX_NAME}' already exists (status={idx.get('status')}).")
                return 0

        collection.create_search_index({"name": INDEX_NAME, "definition": index_definition})
        print(f"Created search index '{INDEX_NAME}'. It may take a moment to become queryable.")
    except OperationFailure as exc:
        print(
            f"Failed to create search index: {exc.details.get('errmsg', str(exc))}",
            file=sys.stderr,
        )
        print(
            "\nIf your deployment does not support create_search_index, create the index "
            "manually in Atlas UI or mongosh using the JSON above.",
            file=sys.stderr,
        )
        return 1
    finally:
        client.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
