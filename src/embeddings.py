import os
import numpy as np
from sentence_transformers import SentenceTransformer

HF_TOKEN = os.getenv('HF_TOKEN')

def embd_chunks(chunks, model_name="all-MiniLM-L6-v2"):

    model = SentenceTransformer(model_name, token=HF_TOKEN)

    texts = [ chunk["text"] for chunk in chunks ]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return model, embeddings


def save_embeddings(
    embeddings,
    path="embeddings.npy"
):
    np.save(path, embeddings)


def load_embeddings(
    path="embeddings.npy"
):
    return np.load(path)