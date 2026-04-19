import chromadb
import json
import os
from datetime import datetime

CHROMA_DATA_DIR = "./chroma_data"
CONTEXT_FILE = "patient_context.json"

# Initialize ChromaDB persistent client
chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_DIR)

# Collections
# We use cosine similarity for DeepFace embeddings
faces_collection = chroma_client.get_or_create_collection(
    name="faces",
    metadata={"hnsw:space": "cosine"}
)

memories_collection = chroma_client.get_or_create_collection(
    name="memories",
    metadata={"hnsw:space": "cosine"}
)

# Objects collection — migrate if dimension mismatch (CLIP 512-dim → DINOv2 384-dim)
def _init_objects_collection():
    col = chroma_client.get_or_create_collection(
        name="objects",
        metadata={"hnsw:space": "cosine"}
    )
    # Check if existing vectors have wrong dimensionality
    if col.count() > 0:
        try:
            sample = col.peek(limit=1)
            embeddings = sample.get("embeddings") if sample else None
            if embeddings is not None and len(embeddings) > 0 and len(embeddings[0]) != 384:
                print(f"[db] Objects collection has {len(sample['embeddings'][0])}-dim vectors, expected 384. Recreating...")
                chroma_client.delete_collection("objects")
                col = chroma_client.get_or_create_collection(
                    name="objects",
                    metadata={"hnsw:space": "cosine"}
                )
                print("[db] Objects collection recreated for DINOv2 384-dim embeddings.")
        except Exception as e:
            print(f"[db] Migration check error: {e}")
    return col

objects_collection = _init_objects_collection()

voices_collection = chroma_client.get_or_create_collection(
    name="voices",
    metadata={"hnsw:space": "cosine"}
)

def add_face(embedding: list[float], metadata: dict):
    import uuid
    face_id = str(uuid.uuid4())
    faces_collection.add(
        embeddings=[embedding],
        metadatas=[metadata],
        ids=[face_id]
    )

def query_face(embedding: list[float], threshold=0.40) -> dict | None:
    """Query faces using top-5 results with majority voting for robust identity."""
    if faces_collection.count() == 0:
        return None
    
    n = min(5, faces_collection.count())
    results = faces_collection.query(
        query_embeddings=[embedding],
        n_results=n
    )
    
    if not ("distances" in results and results["distances"] and results["distances"][0]):
        return None
    
    # Collect all matches within threshold
    from collections import Counter
    name_votes = Counter()
    best_meta = None
    best_dist = float('inf')
    
    for i, dist in enumerate(results["distances"][0]):
        if dist < threshold:
            meta = results["metadatas"][0][i]
            name = meta.get("name", "Unknown")
            name_votes[name] += 1
            if dist < best_dist:
                best_dist = dist
                best_meta = meta
    
    if best_meta:
        return best_meta
    return None

def count_face_vectors(name: str) -> int:
    """Count how many embedding vectors exist for a given person name."""
    data = faces_collection.get(where={"name": name})
    return len(data["ids"]) if data and data["ids"] else 0

def get_patient_context() -> dict:
    if os.path.exists(CONTEXT_FILE):
        with open(CONTEXT_FILE, "r") as f:
            return json.load(f)
    return {
        "routines": [],
        "reminders": [],
        "notes": []
    }

def save_patient_context(context: dict):
    with open(CONTEXT_FILE, "w") as f:
        json.dump(context, f, indent=4)

def add_routine(routine: str):
    ctx = get_patient_context()
    ctx["routines"].append(routine)
    save_patient_context(ctx)

def add_reminder(time: str, task: str):
    ctx = get_patient_context()
    ctx["reminders"].append({"time": time, "task": task})
    save_patient_context(ctx)

def add_note(note: str):
    ctx = get_patient_context()
    ctx["notes"].append(note)
    save_patient_context(ctx)

def build_context_string() -> str:
    ctx = get_patient_context()
    lines = ["Patient Context Information:"]
    
    if ctx["routines"]:
        lines.append("\nRoutines:")
        for r in ctx["routines"]:
            lines.append(f"- {r}")
            
    if ctx["reminders"]:
        lines.append("\nReminders:")
        for r in ctx["reminders"]:
            lines.append(f"- At {r['time']}: {r['task']}")
            
    if ctx["notes"]:
        lines.append("\nNotes:")
        for n in ctx["notes"]:
            lines.append(f"- {n}")

    objects = ctx.get("objects", [])
    if objects:
        lines.append("\nSaved objects in memory:")
        for obj in objects:
            note = obj.get("notes", "")
            name = obj.get("name", "Unknown")
            cat = obj.get("category", "")
            dosage = obj.get("dosage", "")
            parts = [name]
            if cat:
                parts.append(f"({cat})")
            if dosage:
                parts.append(f"— dosage: {dosage}")
            if note:
                parts.append(f"— {note}")
            lines.append(f"- {' '.join(parts)}")
            
    return "\n".join(lines)

def log_session_memory(summary_text: str):
    """
    Saves a text summary of a session to ChromaDB using the default fast local embeddings.
    """
    import uuid
    mem_id = str(uuid.uuid4())
    memories_collection.add(
        documents=[summary_text],
        ids=[mem_id]
    )

def query_session_memories(query_text: str, n_results=3) -> list[str]:
    """
    Queries past session summaries.
    """
    if memories_collection.count() == 0:
        return []
        
    n = min(n_results, memories_collection.count())
    results = memories_collection.query(
        query_texts=[query_text],
        n_results=n
    )
    
    documents = []
    if results and "documents" in results and results["documents"]:
        docs = results["documents"][0]
        for i, doc in enumerate(docs):
            documents.append(doc)
    return documents

def list_faces() -> list[dict]:
    """Return registered people, deduplicated by name, with voice and topic info."""
    data = faces_collection.get()
    seen = {}
    if data and data["ids"]:
        for i, face_id in enumerate(data["ids"]):
            meta = data["metadatas"][i] if data["metadatas"] else {}
            name = meta.get("name", "Unknown")
            if name not in seen:
                seen[name] = {"id": face_id, "name": name, "relation": meta.get("relation", ""), "vectors": 1}
            else:
                seen[name]["vectors"] += 1
    # Enrich with voice status and topics
    for name, info in seen.items():
        info["has_voice"] = count_voice_vectors(name) > 0
        topics_data = get_face_topics(name)
        info["topics"] = topics_data.get("topics", []) if topics_data else []
    return list(seen.values())

def delete_face(face_id: str):
    """Delete ALL vectors for a person. face_id is the name string from the UI."""
    # The frontend sends the name as the face_id since we deduplicate by name
    data = faces_collection.get(where={"name": face_id})
    if data and data["ids"]:
        faces_collection.delete(ids=data["ids"])

def list_notes() -> list[str]:
    """Return all patient notes."""
    ctx = get_patient_context()
    return ctx.get("notes", [])

def delete_note(index: int):
    """Delete a note by its index."""
    ctx = get_patient_context()
    notes = ctx.get("notes", [])
    if 0 <= index < len(notes):
        notes.pop(index)
        ctx["notes"] = notes
        save_patient_context(ctx)

def list_routines() -> list[str]:
    """Return all patient routines."""
    ctx = get_patient_context()
    return ctx.get("routines", [])

def delete_routine(index: int):
    """Delete a routine by its index."""
    ctx = get_patient_context()
    routines = ctx.get("routines", [])
    if 0 <= index < len(routines):
        routines.pop(index)
        ctx["routines"] = routines
        save_patient_context(ctx)

def list_reminders() -> list[dict]:
    """Return all patient reminders."""
    ctx = get_patient_context()
    return ctx.get("reminders", [])

def delete_reminder(index: int):
    """Delete a reminder by its index."""
    ctx = get_patient_context()
    reminders = ctx.get("reminders", [])
    if 0 <= index < len(reminders):
        reminders.pop(index)
        ctx["reminders"] = reminders
        save_patient_context(ctx)

# ========== Object Functions ==========

def add_object_embedding(embedding: list[float], metadata: dict):
    import uuid
    obj_id = str(uuid.uuid4())
    objects_collection.add(
        embeddings=[embedding],
        metadatas=[metadata],
        ids=[obj_id]
    )

def query_object(embedding: list[float], threshold=0.30) -> dict | None:
    """Query objects using top-5 results with majority voting."""
    if objects_collection.count() == 0:
        return None
    
    n = min(5, objects_collection.count())
    results = objects_collection.query(
        query_embeddings=[embedding],
        n_results=n
    )
    
    if not ("distances" in results and results["distances"] and results["distances"][0]):
        return None
    
    from collections import Counter
    name_votes = Counter()
    
    for i, dist in enumerate(results["distances"][0]):
        if dist < threshold:
            meta = results["metadatas"][0][i]
            name = meta.get("name", "Unknown")
            name_votes[name] += 1
    
    if not name_votes:
        return None
    
    # Use the name with the most votes (majority voting)
    winner_name = name_votes.most_common(1)[0][0]
    
    # Return metadata of the closest vector with the winning name
    for i, dist in enumerate(results["distances"][0]):
        if dist < threshold:
            meta = results["metadatas"][0][i]
            if meta.get("name") == winner_name:
                return meta
    
    return None

def count_object_vectors(name: str) -> int:
    data = objects_collection.get(where={"name": name})
    return len(data["ids"]) if data and data["ids"] else 0

def list_objects() -> list[dict]:
    """Return registered objects from patient_context.json with vector counts."""
    ctx = get_patient_context()
    objects = ctx.get("objects", [])
    result = []
    for obj in objects:
        name = obj.get("name", "Unknown")
        result.append({
            **obj,
            "vectors": count_object_vectors(name)
        })
    return result

def delete_object(name: str):
    """Delete all vectors and metadata for an object."""
    data = objects_collection.get(where={"name": name})
    if data and data["ids"]:
        objects_collection.delete(ids=data["ids"])
    delete_object_info(name)

def add_object_info(obj: dict):
    ctx = get_patient_context()
    if "objects" not in ctx:
        ctx["objects"] = []
    ctx["objects"].append(obj)
    save_patient_context(ctx)

def update_object_info(name: str, updates: dict):
    ctx = get_patient_context()
    objects = ctx.get("objects", [])
    for obj in objects:
        if obj.get("name") == name:
            obj.update(updates)
            break
    save_patient_context(ctx)

def delete_object_info(name: str):
    ctx = get_patient_context()
    objects = ctx.get("objects", [])
    ctx["objects"] = [o for o in objects if o.get("name") != name]
    save_patient_context(ctx)

def get_object_info(name: str) -> dict | None:
    ctx = get_patient_context()
    objects = ctx.get("objects", [])
    for obj in objects:
        if obj.get("name") == name:
            return obj
    return None

def objects_count() -> int:
    return objects_collection.count()


# ========== Voice Functions ==========

def add_voice(embedding: list[float], metadata: dict):
    """Store a voice embedding with speaker metadata."""
    import uuid
    voice_id = str(uuid.uuid4())
    voices_collection.add(
        embeddings=[embedding],
        metadatas=[metadata],
        ids=[voice_id]
    )
    print(f"[db] Added voice embedding for '{metadata.get('name', '?')}' (id={voice_id[:8]})")


def query_voice(embedding: list[float], threshold=0.25) -> dict | None:
    """Query voices using top-5 results with majority voting."""
    if voices_collection.count() == 0:
        return None

    n = min(5, voices_collection.count())
    results = voices_collection.query(
        query_embeddings=[embedding],
        n_results=n
    )

    if not ("distances" in results and results["distances"] and results["distances"][0]):
        return None

    from collections import Counter
    name_votes = Counter()
    best_meta = None
    best_dist = float('inf')

    for i, dist in enumerate(results["distances"][0]):
        if dist < threshold:
            meta = results["metadatas"][0][i]
            name = meta.get("name", "Unknown")
            name_votes[name] += 1
            if dist < best_dist:
                best_dist = dist
                best_meta = meta

    if best_meta:
        print(f"[db] Voice match: '{best_meta.get('name')}' (dist={best_dist:.4f}, votes={dict(name_votes)})")
        return best_meta
    return None


def count_voice_vectors(name: str) -> int:
    """Count how many voice embedding vectors exist for a given person."""
    data = voices_collection.get(where={"name": name})
    return len(data["ids"]) if data and data["ids"] else 0


def voices_count() -> int:
    """Total number of voice embeddings in the collection."""
    return voices_collection.count()


def delete_voice(name: str):
    """Delete all voice vectors for a person by name."""
    data = voices_collection.get(where={"name": name})
    if data and data["ids"]:
        voices_collection.delete(ids=data["ids"])
        print(f"[db] Deleted {len(data['ids'])} voice vectors for '{name}'")


# ========== Face Conversation Topic Functions ==========

def update_face_topics(name: str, new_topic: str):
    """Add a conversation topic for a person. Keeps last 10 topics."""
    ctx = get_patient_context()
    convos = ctx.setdefault("face_conversations", {})
    person = convos.setdefault(name, {"topics": [], "last_topic": "", "last_seen": ""})
    person["topics"].append(new_topic)
    person["last_topic"] = new_topic
    person["last_seen"] = datetime.now().isoformat()
    # Keep only the last 10 topics
    person["topics"] = person["topics"][-10:]
    save_patient_context(ctx)
    print(f"[db] Saved topic '{new_topic}' for '{name}' (total: {len(person['topics'])})")


def get_face_topics(name: str) -> dict:
    """Get conversation topics for a person. Returns {} if not found."""
    ctx = get_patient_context()
    return ctx.get("face_conversations", {}).get(name, {})


def delete_face_topic(name: str, topic_index: int) -> bool:
    """Delete a specific conversation topic by index. Returns True if deleted."""
    ctx = get_patient_context()
    convos = ctx.get("face_conversations", {})
    person = convos.get(name)
    if not person or "topics" not in person:
        return False
    topics = person["topics"]
    if 0 <= topic_index < len(topics):
        removed = topics.pop(topic_index)
        person["last_topic"] = topics[-1] if topics else ""
        save_patient_context(ctx)
        print(f"[db] Deleted topic '{removed}' for '{name}' (remaining: {len(topics)})")
        return True
    return False


def list_all_topics() -> dict:
    """Return all face conversation topics. {name: {topics: [...], last_topic: ...}}"""
    ctx = get_patient_context()
    return ctx.get("face_conversations", {})


def has_voice(name: str) -> bool:
    """Check if a person has an enrolled voice."""
    return count_voice_vectors(name) > 0
