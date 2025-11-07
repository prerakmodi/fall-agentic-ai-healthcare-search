# prerequisites
# pip install chromadb sentence-transformers

from sentence_transformers import SentenceTransformer
import chromadb
import uuid
import datetime

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
client = chromadb.PersistentClient(path="./convo-history")

collection = client.get_or_create_collection(
    name="chat_history",
    embedding_function=None
)

def embed_text(text: str):
    vec = embed_model.encode(text, convert_to_numpy=True)
    return vec.tolist()

def store_message(convo_id: str, role: str, text: str):
    emb = embed_text(text)
    # get unique id
    message_id = str(uuid.uuid4())
    metadata = {
        "convo_id": convo_id,
        "message_id": message_id,
        "role": role,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "text": text
    }
    collection.add(
        documents=[text],
        embeddings=[emb],
        metadatas=[metadata],
        ids=[message_id]
    )

# query by specific conversation id and related stuff in it
def retrieve_similar(convo_id: str, text: str, top_k: int = 5):
    query_emb = embed_text(text)
    results = collection.query(
        query_embeddings=[query_emb],
        where = {
            "convo_id":convo_id,
            "role":"user"
        },
        n_results=top_k,
    )
    return results

if __name__ == "__main__":
    convo_id = "session-124"
    store_message(convo_id, "user", "Hello, how are you?")
    store_message(convo_id, "assistant", "I am fine, thank you. What can I help you with?")
    store_message(convo_id, "user", "Tell me about vector databases.")

    new_prompt = "What is a vector database and how can I use it?"
    sims = retrieve_similar(convo_id, new_prompt, top_k=3)
    print("Similar past pieces:", sims["metadatas"])
