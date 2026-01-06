import streamlit as st
from openai import OpenAI
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

#configurations
INDEX_PATH = "aipe_mcgee71_local.faiss"
META_PATH = "aipe_mcgee71_local_meta.json"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # MUST match what you used to build the index

# ----------------------------
# Load the API key from Streamlit secrets
api_key = st.secrets["openai"]["api_key"]

# Create OpenAI client
client = OpenAI(api_key=api_key)

# Load vector store once (cached)
# ----------------------------
@st.cache_resource
def load_vector_assets():
    model = SentenceTransformer(EMBED_MODEL_NAME)
    index = faiss.read_index(INDEX_PATH)
    meta = json.load(open(META_PATH))
    return model, index, meta

embed_model, faiss_index, meta = load_vector_assets()
#--------------------------------



@st.cache_resource
def load_vector_assets():
    model = SentenceTransformer(EMBED_MODEL_NAME)
    index = faiss.read_index(INDEX_PATH)
    meta = json.load(open(META_PATH))
    return model, index, meta

embed_model, faiss_index, meta = load_vector_assets()

def retrieve_lr_rows(query: str, k: int = 8):
    """Local retrieval: query -> local embedding -> FAISS search -> metadata rows."""
    qv = embed_model.encode([query], normalize_embeddings=True)
    qv = np.array(qv, dtype="float32")

    scores, idxs = faiss_index.search(qv, k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        row = meta[idx]
        results.append({**row, "score": float(score)})
    return results

def build_evidence_block(rows):
    """Make a compact evidence block for the LLM."""
    lines = []
    for r in rows:
        lines.append(
            f"- Condition: {r.get('Condition')} | Finding/Maneuver: {r.get('Finding')} | "
            f"LR+: {r.get('LR_plus')} | LR-: {r.get('LR_minus')} | "
            f"Pretest: {r.get('Pretest_low')}-{r.get('Pretest_high')}%"
        )
    return "\n".join(lines)
#----------------------------------------

def PE_recs(clinical_info: str, retrieved_rows: list):
    evidence = build_evidence_block(retrieved_rows)

    prompt = f"""
Clinical scenario (no PHI): {clinical_info}

You must base recommendations primarily on the retrieved likelihood-ratio table evidence below.
If evidence is weak or not clearly relevant, say so and recommend general high-yield exam domains.

Retrieved evidence (likelihood ratios):
{evidence}

Task:
1) Recommend the top 5 specialized physical exam maneuvers/findings to perform next.
2) For each: what to do, what a positive/negative result suggests, and (if available) LR+/LR-.
3) Keep it concise and learner-focused.
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert academic physician focused on medical education. Prioritize evidence from provided LR table."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1200,
        temperature=0.4
    )
    return response.choices[0].message.content
#---------------------------------------------------
#Streamlit app 
st.title("AI-PE: An AI powered tool to help you become an expert at the physical exam")
st.info("""
    This is an educational app that will recommend specialized physical exam tests to perform based on the clinical situation.  
    **Do not include any PHI!**
    """,
    icon="ℹ️"
            
)

st.warning(
"""
**Disclaimer:** This tool is for educational purposes only and should not be used as a clinical decision-making tool.
""",
icon="⚠️"
)
clinical_info = st.text_input("Enter relevant clinical information (no PHI)")

k = st.slider("How many evidence rows to retrieve", min_value=3, max_value=20, value=8)

if st.button("Recommend physical exam maneuvers"):
    if not clinical_info.strip():
        st.error("Please enter clinical information.")
    else:
        with st.spinner("Retrieving evidence from LR table (local)…"):
            retrieved = retrieve_lr_rows(clinical_info, k=k)

        st.subheader("Top retrieved LR-table rows (local)")
        # show a clean table for transparency
        st.dataframe([
            {
                "score": r["score"],
                "Condition": r.get("Condition"),
                "Finding/Maneuver": r.get("Finding"),
                "LR+": r.get("LR_plus"),
                "LR-": r.get("LR_minus"),
                "Pretest low": r.get("Pretest_low"),
                "Pretest high": r.get("Pretest_high"),
            }
            for r in retrieved
        ])

        with st.spinner("Generating learner-focused recommendations…"):
            recs = PE_recs(clinical_info, retrieved)

        st.subheader("Recommendations")
        st.write(recs)