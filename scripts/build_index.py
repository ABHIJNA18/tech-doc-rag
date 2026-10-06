# ---------------------------------------------------------
# build_index.py
#
# Builds the vector index for retrieval from chunks.jsonl.
#
# Steps:
#   1. Load chunks and convert them into LangChain Documents
#      (page_content = chunk text with header, metadata = citation
#      fields).
#   2. Rebuild the Chroma collection from scratch: embed every
#      Document with OpenAI (via the embedding cache) and store
#      vector + text + metadata, with chunk_id as the ID.
#   3. Verify: number and IDs of stored vectors, and a few test
#      questions that must find the expected pages.
#   4. Write manifest.json next to the index: what it was built from
#      (chunks checksum, git commit) and with which model.
#
# Needs OPENAI_API_KEY in .env (see .env.example).
#
# Usage:
#   python scripts/build_index.py
#   python scripts/build_index.py --chunks path/to/chunks.jsonl
#   python scripts/build_index.py --convert-only     (no OpenAI calls)
# ---------------------------------------------------------

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

# Make src/ importable when running this file as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag import config  # noqa: E402
from rag.documents import chunk_to_document, document_to_chunk, load_chunks  # noqa: E402
from rag.vectorstore import DISTANCE, chroma_client, make_embeddings, open_vector_store  # noqa: E402

# Test questions and a part of the URL of the page that must be among
# the top results. A quick sanity check, not the evaluation.
TEST_QUERIES = [
    ("What does a 429 error mean in the AI Model Hub?", "ai-model-hub/error-codes"),
    ("What is the maximum number of replicas in an In-Memory DB replica set?", "in-memory-db"),
    ("What is the maximum page size when listing In-Memory DB replica sets?", "retrieve-all-replicaset"),
    ("What is ACPI?", "glossary-of-terms"),
    ("What happens if I forget my backup encryption password?", "backup-service-faqs"),
    ("Which fields do I need to send to create a VM Auto Scaling group?", "create-a-vm-auto-scaling-group"),
]

TOP_K = 5


# ---------------------------------------------------------
# Step 1: Documents
# ---------------------------------------------------------


def check_documents(chunks: list[dict], documents: list) -> list[str]:
    """
    Sanity checks on the conversion. Returns a list of problems.
    """

    problems = []

    ids = [document.id for document in documents]
    if len(set(ids)) != len(ids):
        problems.append("duplicate document IDs")

    for chunk, document in zip(chunks, documents):

        if document.page_content != chunk["text"]:
            problems.append(f"{chunk['chunk_id']}: page_content differs from chunk text")

        # Chroma accepts only simple metadata values.
        for key, value in document.metadata.items():
            if not isinstance(value, (str, int, float, bool)):
                problems.append(f"{chunk['chunk_id']}: metadata {key} is {type(value).__name__}")

        # Converting back must give the original record.
        if document_to_chunk(document) != chunk:
            problems.append(f"{chunk['chunk_id']}: round trip changed the record")

    return problems


# ---------------------------------------------------------
# Step 2: Embed and store
# ---------------------------------------------------------


def cached_vector_count() -> int:
    """
    Number of vectors in the embedding cache (one file each).
    """

    if not config.EMBEDDING_CACHE_DIR.exists():
        return 0

    return sum(1 for path in config.EMBEDDING_CACHE_DIR.rglob("*") if path.is_file())


def rebuild_collection(documents: list) -> None:
    """
    Delete the old collection (no stale vectors survive) and add all
    documents. Adding is where embedding happens: each text is
    embedded (from the cache, or by OpenAI) and stored with its
    metadata.
    """

    client = chroma_client()

    if config.COLLECTION_NAME in [collection.name for collection in client.list_collections()]:
        client.delete_collection(config.COLLECTION_NAME)

    vector_store = open_vector_store(make_embeddings(cached=True), client)
    vector_store.add_documents(documents, ids=[document.id for document in documents])


# ---------------------------------------------------------
# Step 3: Verify
# ---------------------------------------------------------


def verify_collection(documents: list) -> tuple[list[str], int]:
    """
    The stored IDs must be exactly the document IDs. Returns the
    problems and the vector dimension.
    """

    collection = chroma_client().get_collection(config.COLLECTION_NAME)
    stored = collection.get(include=["embeddings"])

    problems = []

    if sorted(stored["ids"]) != sorted(document.id for document in documents):
        problems.append(f"stored IDs differ from documents ({len(stored['ids'])} stored)")

    dimension = len(stored["embeddings"][0]) if len(stored["ids"]) else 0

    return problems, dimension


def run_test_queries() -> int:
    """
    Search each test question and show the top results. Lower cosine
    distance = more similar. Returns how many found the expected page.
    """

    vector_store = open_vector_store(make_embeddings(cached=False))
    found = 0

    for question, expected in TEST_QUERIES:
        results = vector_store.similarity_search_with_score(question, k=TOP_K)
        rank = next(
            (i for i, (document, _) in enumerate(results, 1) if expected in document.metadata["source_url"]),
            None,
        )
        found += rank is not None

        print(f"\nQ: {question}")
        print(f"   expected page '{expected}': {'rank ' + str(rank) if rank else 'NOT in top ' + str(TOP_K)}")
        for i, (document, distance) in enumerate(results[:3], 1):
            header = document.page_content.split("\n", 1)[0]
            print(f"   {i}. {distance:.3f}  {header[:100]}")

    return found


# ---------------------------------------------------------
# Step 4: Manifest
# ---------------------------------------------------------


def git_commit() -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=config.REPO_DIR, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, cwd=config.REPO_DIR, check=True
        ).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_manifest(chunks_file: Path, document_count: int, dimension: int) -> Path:
    """
    Record what this index was built from and with, so evaluation
    results can always be traced back to an exact index.
    """

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "chunks_file": str(chunks_file),
        "chunks_sha256": hashlib.sha256(chunks_file.read_bytes()).hexdigest(),
        "documents": document_count,
        "collection": config.COLLECTION_NAME,
        "embedding_model": config.EMBEDDING_MODEL,
        "embedding_dimension": dimension,
        "distance": DISTANCE,
        "versions": {package: version(package) for package in ("chromadb", "langchain-chroma", "langchain-openai")},
    }

    path = config.CHROMA_DIR / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return path


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------


def main():

    parser = argparse.ArgumentParser(description="Build the Chroma index from chunks.jsonl")
    parser.add_argument("--chunks", type=Path, default=config.CHUNKS_FILE)
    parser.add_argument("--convert-only", action="store_true", help="only convert and check, no OpenAI calls")
    args = parser.parse_args()

    # Step 1
    chunks = load_chunks(args.chunks)
    documents = [chunk_to_document(chunk) for chunk in chunks]
    problems = check_documents(chunks, documents)

    print("=" * 50)
    print("1. Chunks -> LangChain Documents")
    print("=" * 50)
    print(f"Chunks file: {args.chunks}")
    print(f"Documents:   {len(documents)}")
    print(f"Checks:      {'OK' if not problems else f'{len(problems)} problems'}")
    for problem in problems[:10]:
        print(f"  - {problem}")

    if problems:
        sys.exit("Fix the conversion problems before indexing.")

    if args.convert_only:
        return

    # Step 2
    print("\n" + "=" * 50)
    print("2. Embed and store in Chroma")
    print("=" * 50)

    cached_before = cached_vector_count()
    rebuild_collection(documents)
    newly_embedded = cached_vector_count() - cached_before

    print(f"Collection:  {config.COLLECTION_NAME} in {config.CHROMA_DIR}")
    print(f"Model:       {config.EMBEDDING_MODEL}")
    print(f"Embedded by OpenAI: {newly_embedded}   from cache: {len(documents) - newly_embedded}")

    # Step 3
    print("\n" + "=" * 50)
    print("3. Verify")
    print("=" * 50)

    problems, dimension = verify_collection(documents)
    print(f"Vectors stored: {len(documents) if not problems else 'MISMATCH'}   dimension: {dimension}")
    for problem in problems:
        print(f"  - {problem}")

    found = run_test_queries()
    print(f"\nTest questions with the expected page in the top {TOP_K}: {found}/{len(TEST_QUERIES)}")

    # Step 4
    manifest_path = write_manifest(args.chunks, len(documents), dimension)
    print(f"\nManifest: {manifest_path}")


if __name__ == "__main__":
    main()
