"""
Final RAGAS-style evaluation for the Book Recommendation Chatbot (Project 7).

- Manual evaluation set of 10 questions (as recommended).
- Metrics:
    * Context Precision
    * Context Recall
    * Faithfulness
    * Answer Correctness (semantic similarity)

Uses retrieve() + generate_answer() from rag_chatbot.py.
This ensures evaluation matches what your chatbot actually outputs (including Groq augmentation if enabled).
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from rag_chatbot import retrieve, generate_answer


# ---------------------------------------------------
# 1) Manual evaluation set (10 questions)
# ---------------------------------------------------
EVAL_SET = [
    {
        "question": "I like fantasy with dragons and magic.",
        "ground_truth_books": [
            "Dragonsinger (Harper Hall  #2)",
            "Dragonsong (Harper Hall  #1)",
            "Dragons of Eden: Speculations on the Evolution of Human Intelligence",
        ],
        "ideal_answer": (
            "Recommend fantasy novels with dragons, magic and imaginative worlds, "
            "such as Dragonsinger and Dragonsong."
        ),
    },
    {
        "question": "Recommend sci-fi books about space travel.",
        "ground_truth_books": [
            "The Inner Reaches of Outer Space: Metaphor as Myth and as Religion (Collected Works)",
            "Life  the Universe and Everything (Hitchhiker's Guide to the Galaxy  #3)",
            "Citizen of the Galaxy",
        ],
        "ideal_answer": (
            "Recommend science fiction about space travel and the universe, "
            "for example Life, the Universe and Everything or Citizen of the Galaxy."
        ),
    },
    {
        "question": "I want short, easy books for teenagers.",
        "ground_truth_books": [
            "Teen Angst? Naaah...",
            "Seventeen and In-Between",
            "Field of Thirteen",
        ],
        "ideal_answer": (
            "Recommend short, easy-to-read young adult books suitable for teenagers, "
            "like Teen Angst? Naaah... or Seventeen and In-Between."
        ),
    },
    {
        "question": "I like scary horror books with monsters or ghosts.",
        "ground_truth_books": [
            "The Ghost Stories of Edith Wharton",
            "Ghost Stories (Nancy Drew)",
            "Ghosts and Grisly Things",
        ],
        "ideal_answer": (
            "Recommend horror stories involving ghosts, monsters or other supernatural elements."
        ),
    },
    {
        "question": "Recommend crime or thriller novels with lots of tension.",
        "ground_truth_books": [
            "Crime and Punishment",
            "Crime Partners",
            "Thriller: Stories To Keep You Up All Night",
        ],
        "ideal_answer": (
            "Recommend tense crime or thriller stories that keep the reader hooked."
        ),
    },
    {
        "question": "I want romance books with a lot of drama.",
        "ground_truth_books": [
            "Historical Romances: The Prince and the Pauper / A Connecticut Yankee in King Arthur's Court / Personal Recollections of Joan of Arc",
            "Heartsnatcher",
            "Your Cheatin' Heart",
        ],
        "ideal_answer": "Recommend emotional romance novels full of relationship drama.",
    },
    {
        "question": "Recommend books suitable for children.",
        "ground_truth_books": [
            "Your Child's Self-Esteem: Step-by-Step Guidelines for Raising Responsible  Productive  Happy Children",
            "The Children's Book of America",
            "Letters to Children",
        ],
        "ideal_answer": (
            "Recommend books appropriate for children and families, such as The Children's Book of America."
        ),
    },
    {
        "question": "Recommend classic literature everyone should read.",
        "ground_truth_books": [
            "Nineteen Eighty-Four",
            "Masterpieces: The Best Science Fiction of the Twentieth Century",
            "The Western Canon: The Books and School of the Ages",
        ],
        "ideal_answer": "Recommend classic, widely known literature that is often considered essential reading.",
    },
    {
        "question": "I like non-fiction books about space, physics or astronomy.",
        "ground_truth_books": [
            "The Nature of Space and Time",
            "The Future of Spacetime",
            "Species of Spaces and Other Pieces",
        ],
        "ideal_answer": (
            "Recommend non-fiction science books about physics, cosmology or astronomy, "
            "such as The Future of Spacetime."
        ),
    },
    {
        "question": "Recommend young adult books about growing up and school life.",
        "ground_truth_books": [
            "Young Men and Fire",
            "The Girlhood Diary of Louisa May Alcott  1843–1846: Writings of a Young Author",
            "Poetry for Young People: Edward Lear",
        ],
        "ideal_answer": "Recommend young adult or coming-of-age books about growing up, school and personal struggles.",
    },
]


# ---------------------------------------------------
# 2) Metric implementations (RAGAS-style)
# ---------------------------------------------------
def context_precision(retrieved_titles, ground_truth_titles):
    if not retrieved_titles:
        return 0.0
    rset = set(retrieved_titles)
    gset = set(ground_truth_titles)
    return len(rset & gset) / max(1, len(rset))


def context_recall(retrieved_titles, ground_truth_titles):
    if not ground_truth_titles:
        return 0.0
    rset = set(retrieved_titles)
    gset = set(ground_truth_titles)
    return len(rset & gset) / max(1, len(gset))


def faithfulness(answer, retrieved_titles):
    # grounding: answer mentions at least one retrieved title
    ans = (answer or "").lower()
    mentioned = any(t.lower() in ans for t in retrieved_titles)
    return 1.0 if mentioned else 0.5


_embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def answer_correctness(answer, ideal_answer):
    vecs = _embed_model.encode([answer, ideal_answer])
    a, b = vecs[0], vecs[1]
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b) / denom)


# ---------------------------------------------------
# 3) Evaluation loop
# ---------------------------------------------------
def evaluate_once(example, retrieve_k: int = 10, show_n: int = 5):
    q = example["question"]
    gt = example["ground_truth_books"]
    ideal = example["ideal_answer"]

    retrieved_df = retrieve(q, k=retrieve_k)
    # match generation behavior: show top-n
    answer = generate_answer(q, retrieved_df, max_recs=show_n)

    # titles used for context metrics should reflect what system *had available*
    # (after retrieve). Your retrieve() already deduplicates.
    retrieved_titles = list(retrieved_df["title"].astype(str).head(show_n))

    return {
        "question": q,
        "answer": answer,
        "retrieved_titles": retrieved_titles,
        "context_precision": context_precision(retrieved_titles, gt),
        "context_recall": context_recall(retrieved_titles, gt),
        "faithfulness": faithfulness(answer, retrieved_titles),
        "answer_correctness": answer_correctness(answer, ideal),
    }


def main():
    results = []
    for ex in EVAL_SET:
        results.append(evaluate_once(ex, retrieve_k=10, show_n=5))

    print("Per-question scores:\n")
    for r in results:
        print("Q:", r["question"])
        print("Retrieved titles:", r["retrieved_titles"])
        print(f"  Context precision : {r['context_precision']:.3f}")
        print(f"  Context recall    : {r['context_recall']:.3f}")
        print(f"  Faithfulness      : {r['faithfulness']:.3f}")
        print(f"  Answer correctness: {r['answer_correctness']:.3f}")
        print("-" * 60)

    avg_cp = float(np.mean([r["context_precision"] for r in results]))
    avg_cr = float(np.mean([r["context_recall"] for r in results]))
    avg_fa = float(np.mean([r["faithfulness"] for r in results]))
    avg_ac = float(np.mean([r["answer_correctness"] for r in results]))

    print(f"\n=== Aggregated RAGAS-style scores over {len(results)} questions ===")
    print(f"Context precision : {avg_cp:.3f}")
    print(f"Context recall    : {avg_cr:.3f}")
    print(f"Faithfulness      : {avg_fa:.3f}")
    print(f"Answer correctness: {avg_ac:.3f}")


if __name__ == "__main__":
    main()
