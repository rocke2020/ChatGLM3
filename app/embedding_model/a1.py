from icecream import ic
from sentence_transformers import SentenceTransformer

ic.configureOutput(includeContext=True, argToStringFunction=str)
ic.lineWrapWidth = 120

EMBEDDING_PATH = "/mnt/nas1/models/BAAI/bge-m3"

sentences_1 = ["What is BGE M3?", "Defination of BM25"]
sentences_2 = [
    "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
    "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document",
]


def test():
    """the embeddings are the same with or without normalization for bge-m3.
    embeddings_1.shape: (batch-size, 1024), ndarray
    """
    model = SentenceTransformer(EMBEDDING_PATH, device="cuda")
    embeddings_1 = model.encode(sentences_1, normalize_embeddings=True)
    embeddings_2 = model.encode(sentences_2, normalize_embeddings=True)
    ic(embeddings_1.shape, embeddings_1[:20])
    similarity = embeddings_1 @ embeddings_2.T
    print(similarity)
    embeddings_11 = model.encode(sentences_1, normalize_embeddings=False)
    embeddings_21 = model.encode(sentences_2, normalize_embeddings=False)
    ic(embeddings_11.shape)
    ic(embeddings_11[:20])
    similarity = embeddings_11 @ embeddings_21.T
    print(similarity)
    ic(type(embeddings_1))

test()
