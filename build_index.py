"""
Index building (retrieval-based database) for Project 7.

This script:
  - loads clean_books.csv
  - builds dense embeddings over "title + authors + description"
    using DPRContextEncoder
  - saves:
      * book_embeddings.npy      (retriever index)
      * books_metadata.csv       (metadata for each book)
"""

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer


def main():
    df = pd.read_csv("clean_books.csv")

    texts = (
        df["title"].fillna("") + " - " +
        df["authors"].fillna("") + " - " +
        df["description"].fillna("")
    ).tolist()

    tokenizer = DPRContextEncoderTokenizer.from_pretrained(
        "facebook/dpr-ctx_encoder-single-nq-base"
    )
    model = DPRContextEncoder.from_pretrained(
        "facebook/dpr-ctx_encoder-single-nq-base"
    )
    model.eval()

    embeddings = []
    for t in tqdm(texts, desc="Encoding books with DPR"):
        inputs = tokenizer(
            t,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        )
        with torch.no_grad():
            emb = model(**inputs).pooler_output[0].cpu().numpy()
        embeddings.append(emb)

    emb = np.vstack(embeddings).astype("float32")
    np.save("book_embeddings.npy", emb)
    df.to_csv("books_metadata.csv", index=False)

    print("✅ Saved book_embeddings.npy with shape:", emb.shape)
    print("✅ Saved books_metadata.csv")


if __name__ == "__main__":
    main()
