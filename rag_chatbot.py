"""
RAG-based Book Recommendation Chatbot (Project 7)

Retriever:
  - DPRQuestionEncoder over synthetic book descriptions.
  - PLUS a lightweight keyword filter to reduce off-topic books.

Generator:
  - Deterministic top-N list: selects the best books among retrieved
    and prints a numbered recommendation list.

✅ Add-on:
  - Optional Groq usage to produce ONLY:
      (a) genre label
      (b) unique short description (1–2 sentences)
    for each recommended book.
  - If GROQ_API_KEY is not set, we fall back to dataset description.
"""

import os
import re
import json
import numpy as np
import torch
import pandas as pd
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

# -------------------------
# Optional Groq integration
# -------------------------
USE_GROQ = True  # set False if you want to disable Groq quickly
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

def _groq_client():
    """
    Lazy import so the file still runs without Groq installed.
    """
    try:
        from groq import Groq
        return Groq(api_key=os.getenv("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"))
    except Exception:
        return None

def _safe_text(x, max_len=1200):
    x = "" if x is None else str(x)
    x = re.sub(r"\s+", " ", x).strip()
    return x[:max_len]

def groq_genre_and_description(book_row: pd.Series, user_query: str) -> dict:
    """
    Returns: {"genre": "...", "description": "..."}
    If Groq not available, returns fallback based on dataset.
    """
    # Fallback if Groq is disabled or key missing
    if not USE_GROQ or not os.getenv("GROQ_API_KEY"):
        return {
            "genre": "Recommendation",
            "description": _safe_text(book_row.get("description", ""), 260) or "No description available.",
        }

    client = _groq_client()
    if client is None:
        return {
            "genre": "Recommendation",
            "description": _safe_text(book_row.get("description", ""), 260) or "No description available.",
        }

    title = _safe_text(book_row.get("title", "Unknown title"), 120)
    authors = _safe_text(book_row.get("authors", "Unknown author"), 120)
    rating = book_row.get("average_rating", None)
    rating_str = "N/A"
    try:
        if rating is not None and not pd.isna(rating):
            rating_str = f"{float(rating):.2f}/5"
    except Exception:
        pass

    raw_desc = _safe_text(book_row.get("description", ""), 900)
    query = _safe_text(user_query, 220)

    system = (
        "You are helping a book recommendation app. "
        "You must output STRICT JSON only."
    )

    prompt = f"""
User request: "{query}"

Book metadata:
- Title: "{title}"
- Author(s): "{authors}"
- Rating: {rating_str}
- Original description (may be noisy): "{raw_desc}"

Task:
Create ONLY:
1) A short genre label (1–2 words), e.g. "Sci-Fi", "Fantasy", "Romance", "Mystery", "Horror", "Non-fiction", "Classic", "YA".
2) A UNIQUE short description (1–2 sentences, max 45 words) that fits the user's request and does NOT copy the original description.

Rules:
- Do NOT repeat the title in the description.
- Do NOT add extra keys.
- Output must be valid JSON with exactly keys: genre, description.
"""

    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=140,
        )
        content = resp.choices[0].message.content.strip()

        # Try to parse JSON strictly
        data = json.loads(content)

        genre = _safe_text(data.get("genre", "Recommendation"), 30)
        desc = _safe_text(data.get("description", ""), 260)

        if not desc:
            desc = _safe_text(book_row.get("description", ""), 260) or "No description available."

        return {"genre": genre or "Recommendation", "description": desc}

    except Exception:
        # Fallback on any error
        return {
            "genre": "Recommendation",
            "description": _safe_text(book_row.get("description", ""), 260) or "No description available.",
        }

# -------------------------------------------------
# Load metadata and DPR index (retrieval database)
# -------------------------------------------------
df = pd.read_csv("books_metadata.csv")
embeddings = np.load("book_embeddings.npy")
embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)

# Lowercased title/description for keyword filter
TITLE_LOWER = df["title"].fillna("").astype(str).str.lower()
DESC_LOWER = df["description"].fillna("").astype(str).str.lower()

# -------------------------------------------------
# Load DPR question encoder (retriever side)
# -------------------------------------------------
q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
    "facebook/dpr-question_encoder-single-nq-base"
)
q_model = DPRQuestionEncoder.from_pretrained(
    "facebook/dpr-question_encoder-single-nq-base"
)
q_model.eval()

# -------------------------------------------------
# Simple keyword mapping for candidate filtering
# -------------------------------------------------
KEYWORD_GROUPS = {
    "fantasy": ["fantasy", "dragon", "dragons", "magic", "wizard", "sword"],
    "sci-fi": ["sci fi", "scifi", "science fiction", "space", "galaxy", "planet", "alien", "robot"],
    "romance": ["romance", "romantic", "love", "heart"],
    "mystery": ["mystery", "detective", "crime"],
    "thriller": ["thriller", "suspense", "spy"],
    "horror": ["horror", "ghost", "vampire"],
    "children": ["children", "child", "kid", "kids"],
    "teen": ["teen", "young adult", "ya"],
    "history": ["history", "historical", "war"],
}

def _get_candidate_indices(query: str, min_candidates: int = 30):
    q_low = query.lower()
    matched_tokens = set()

    for group_words in KEYWORD_GROUPS.values():
        for w in group_words:
            if w in q_low:
                matched_tokens.add(w)

    if not matched_tokens:
        return np.arange(len(df))

    mask = pd.Series(False, index=df.index)
    for w in matched_tokens:
        mask |= TITLE_LOWER.str.contains(w, regex=False) | DESC_LOWER.str.contains(w, regex=False)

    candidate_idx = np.where(mask.values)[0]

    if len(candidate_idx) < min_candidates:
        return np.arange(len(df))

    return candidate_idx

def retrieve(query: str, k: int = 10) -> pd.DataFrame:
    """
    Dense retrieval using DPR + keyword-based candidate filtering.
    ✅ Deduplicate by title (best DPR score kept)
    """
    cand_idx = _get_candidate_indices(query)
    cand_emb = embeddings_norm[cand_idx]

    inputs = q_tokenizer(
        query,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )
    with torch.no_grad():
        q_emb = q_model(**inputs).pooler_output[0].cpu().numpy()

    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-12)
    scores = cand_emb @ q_emb

    pre_k = min(len(scores), max(k * 6, 60))
    top_local = np.argpartition(-scores, pre_k - 1)[:pre_k]
    top_local = top_local[np.argsort(-scores[top_local])]

    top_global = cand_idx[top_local]

    out = df.iloc[top_global].copy()
    out["_score"] = scores[top_local]

    out["title"] = out["title"].fillna("").astype(str)
    out = out.sort_values("_score", ascending=False)
    out = out.drop_duplicates(subset=["title"], keep="first")

    return out.head(k).drop(columns=["_score"], errors="ignore")

def generate_answer(user_query: str, retrieved_books: pd.DataFrame, max_recs: int = 5) -> str:
    """
    Deterministic generator + optional Groq rewrite (genre + unique short description only).
    """
    if retrieved_books.empty:
        return "I couldn't find any matching books in the database, sorry."

    retrieved_books = retrieved_books.copy()
    retrieved_books["title"] = retrieved_books["title"].fillna("").astype(str)
    retrieved_books = retrieved_books.drop_duplicates(subset=["title"], keep="first")

    if "average_rating" in retrieved_books.columns:
        retrieved_books = retrieved_books.sort_values("average_rating", ascending=False)

    retrieved_books = retrieved_books.head(max_recs)

    lines = []
    lines.append("Here are some books I recommend based on what you said:\n")

    for i, (_, row) in enumerate(retrieved_books.iterrows(), start=1):
        title = str(row.get("title", "Unknown title"))
        authors = str(row.get("authors", "Unknown author"))
        rating = row.get("average_rating", None)
        rating_str = (
            f"{float(rating):.2f}/5"
            if isinstance(rating, (int, float, np.floating)) and not pd.isna(rating)
            else "N/A"
        )

        # ✅ Groq output (genre + unique description only)
        gd = groq_genre_and_description(row, user_query)
        genre = gd["genre"]
        desc = gd["description"]

        line = f"{i}. {title} by {authors} (rating {rating_str}) [{genre}] – {desc}"
        lines.append(line)

    return "\n".join(lines)

def chat():
    print("📚 BookBot Ready! Type 'quit' to exit.")
    print("Try queries like: 'I like fantasy with dragons and magic' "
          "or 'recommend sci-fi books about space travel'.")
    while True:
        try:
            user_input = input("\nUser: ")
        except EOFError:
            break

        if not user_input.strip():
            continue

        if user_input.strip().lower() in {"quit", "exit"}:
            break

        retrieved = retrieve(user_input, k=10)
        answer = generate_answer(user_input, retrieved, max_recs=5)
        print("\nBookBot:", answer)

if __name__ == "__main__":
    chat()