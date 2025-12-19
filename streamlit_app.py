import streamlit as st
import pandas as pd

from rag_chatbot import retrieve, groq_genre_and_description

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="BookBot", page_icon="📚", layout="wide")

# ---------------------------
# Simple clean CSS
# ---------------------------
st.markdown(
    """
    <style>
    .main { background: #0b1220; color: #e5e7eb; }
    .block-container { max-width: 1050px; padding-top: 1.75rem !important; padding-bottom: 2rem !important; }

    h1,h2,h3 { color: #e5e7eb; }
    .subtle { color: #9ca3af; font-size: 0.95rem; margin-top: -0.2rem; }

    .card {
        background: #0f172a;
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 14px;
        padding: 14px 16px;
        margin: 12px 0;
    }
    .title { font-weight: 700; font-size: 1.05rem; margin-bottom: 4px; }
    .meta { color: #9ca3af; font-size: 0.9rem; margin-bottom: 10px; }
    .desc { color: #e5e7eb; font-size: 0.95rem; line-height: 1.45; white-space: pre-wrap; }

    .pill {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        border: 1px solid rgba(148, 163, 184, 0.22);
        color: #cbd5e1;
        margin-right: 6px;
        margin-top: 10px;
    }

    /* nicer input + button */
    div[data-baseweb="input"] input { border-radius: 10px !important; }
    button[kind="primary"] { border-radius: 10px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Helpers
# ---------------------------
def stars_for_rating(r):
    if r is None or pd.isna(r):
        return "No rating"
    val = float(r)
    full = int(round(val))
    full = max(0, min(full, 5))
    return "⭐" * full + f"  ({val:.2f}/5)"

def dedupe_and_sort(df: pd.DataFrame, show_n: int) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    out["title"] = out["title"].fillna("").astype(str)

    # rating sort (like your generator)
    if "average_rating" in out.columns:
        out = out.sort_values("average_rating", ascending=False, na_position="last")

    # remove duplicates by title (extra safety)
    out = out.drop_duplicates(subset=["title"], keep="first")

    return out.head(show_n)

def run_search():
    q = st.session_state.get("query", "").strip()
    if not q:
        return

    show_n = int(st.session_state.get("show_n", 5))
    # retrieve a bit more so sorting+dedupe still yields enough
    retrieve_k = max(10, show_n * 3)

    retrieved_df = retrieve(q, k=retrieve_k)
    final_df = dedupe_and_sort(retrieved_df, show_n=show_n)

    # Precompute Groq genre+desc for UI so it matches terminal output.
    # (This does NOT change retrieval; only enriches display.)
    genres = []
    descs = []
    for _, row in final_df.iterrows():
        gd = groq_genre_and_description(row, q)
        genres.append(gd.get("genre", "Recommendation"))
        descs.append(gd.get("description", ""))

    final_df = final_df.copy()
    final_df["llm_genre"] = genres
    final_df["llm_description"] = descs

    st.session_state["last_query"] = q
    st.session_state["last_df"] = final_df

# ---------------------------
# State
# ---------------------------
if "last_query" not in st.session_state:
    st.session_state["last_query"] = ""
if "last_df" not in st.session_state:
    st.session_state["last_df"] = pd.DataFrame()

# ---------------------------
# Sidebar (only what you asked)
# ---------------------------
with st.sidebar:
    st.markdown("### Settings")
    st.slider("How many recommendations?", 3, 10, 5, step=1, key="show_n")

# ---------------------------
# Main UI
# ---------------------------
st.markdown("# 📚 BookBot")
st.markdown(
    "<div class='subtle'>Type your request and press <b>Enter</b> (or click Find books).</div>",
    unsafe_allow_html=True,
)
st.markdown("---")

st.text_input(
    "What do you feel like reading?",
    placeholder="Example: I like history, ancient Rome, non-fiction",
    key="query",
    on_change=run_search,   # ✅ Enter triggers search
)

if st.button("🔎 Find books", use_container_width=True):
    run_search()

st.markdown("---")

# ---------------------------
# Results
# ---------------------------
q = st.session_state["last_query"]
df = st.session_state["last_df"]

if q and df is not None and not df.empty:
    st.markdown(f"## Results for: _{q}_")

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        title = str(row.get("title", "Unknown title"))
        authors = str(row.get("authors", "Unknown author"))
        rating = row.get("average_rating", None)

        genre = str(row.get("llm_genre", "Recommendation")).strip()
        desc = str(row.get("llm_description", "")).strip()

        # fallback if empty
        if not desc:
            desc = str(row.get("description", "")).replace("\\n", " ").strip()

        language = str(row.get("language_code", "")).strip()
        num_pages = row.get("num_pages", None)

        pills = []
        if genre:
            pills.append(f"<span class='pill'>{genre}</span>")
        if language:
            pills.append(f"<span class='pill'>LANG: {language}</span>")
        if num_pages is not None and not pd.isna(num_pages):
            pills.append(f"<span class='pill'>{int(num_pages)} pages</span>")

        st.markdown(
            f"""
            <div class="card">
              <div class="title">{i}. {title}</div>
              <div class="meta">by {authors} • {stars_for_rating(rating)}</div>
              <div class="desc">{desc}</div>
              {''.join(pills)}
            </div>
            """,
            unsafe_allow_html=True,
        )

elif q:
    st.info("No matching books found.")
else:
    st.info("Type a query and press Enter.")
