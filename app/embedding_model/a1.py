from icecream import ic
from sentence_transformers import SentenceTransformer
import torch
from torch import linalg as LA
from numpy.linalg import norm


ic.configureOutput(includeContext=True, argToStringFunction=str)
ic.lineWrapWidth = 120

EMBEDDING_PATH = "/mnt/nas1/models/BAAI/bge-m3"

sentences_1 = ["What is BGE M3?", "Defination of BM25"]
sentences_2 = [
    "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
    "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document",
]
sentences_3 = [
    "中国人在纽约.",
    "三八妇女节是国际性的节日, 是妇女的节日.",
]


def test():
    """the embeddings are the same with or without normalization for bge-m3.
    embeddings_1.shape: (batch-size, 1024), ndarray
    """
    model = SentenceTransformer(EMBEDDING_PATH, device="cuda")
    embeddings_1 = model.encode(sentences_1, normalize_embeddings=True)
    embeddings_2 = model.encode(sentences_2, normalize_embeddings=True)

    ic(embeddings_1.shape, embeddings_1[:20])
    ic(norm(embeddings_1, axis=1), norm(embeddings_2, axis=1))

    similarity = embeddings_1 @ embeddings_2.T
    print(similarity)
    embeddings_11 = model.encode(sentences_1, normalize_embeddings=False)
    embeddings_21 = model.encode(sentences_2, normalize_embeddings=False)
    ic(embeddings_11.shape)
    ic(embeddings_11[:20])
    ic(norm(embeddings_11, axis=1), norm(embeddings_21, axis=1))
    similarity = embeddings_11 @ embeddings_21.T
    print(similarity)
    ic(type(embeddings_1))
    r = model.encode(sentences_3, normalize_embeddings=False)
    ic(norm(r, axis=1))


test()
