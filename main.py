import streamlit as st
from openai import OpenAI
import json
import numpy as np

from sentence_transformers import SentenceTransformer

try:
    import faiss  # type: ignore
except ModuleNotFoundError:
    faiss = None

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

def generate_differential(clinical_info: str, n: int = 8):
    """
    Call #1: produce a broad differential in strict JSON for parsing.
    Returns: dict with keys: differential (list[str])
    """
    prompt = f"""
Clinical scenario (no PHI):
{clinical_info}

Task:
Generate a BROAD but plausible differential diagnosis list. You are an academic educational attending teaching medical residents. 

Return STRICT JSON ONLY with this schema:
{{
  "differential": ["dx1", "dx2", "..."]
}}

Rules:
- differential should have {n} to {n+4} items
- do not include management or treatment
"""

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert academic physician. Return strict JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=500,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)

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

def build_retrieval_query(clinical_info: str, ddx: dict) -> str:
    dx_list = ddx.get("differential", [])
    # Keep it compact; embeddings like shorter, information-dense text.
    dx_str = ", ".join(dx_list[:10])
    return f"{clinical_info}\nLikely diagnoses: {dx_str}"

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

def build_dx_queries(ddx: dict, max_dx: int = 8):
    dx_list = ddx.get("differential", [])[:max_dx]
    return [f"{dx} physical exam finding likelihood ratio" for dx in dx_list]

def retrieve_lr_rows_multi(queries, k_each=4):
    merged = {}
    for q in queries:
        rows = retrieve_lr_rows(q, k=k_each)
        for r in rows:
            # stable key: use whatever is unique in your meta; here a tuple
            key = (r.get("Condition"), r.get("Finding"))
            if key not in merged or r["score"] > merged[key]["score"]:
                merged[key] = r
    # sort
    return sorted(merged.values(), key=lambda x: x["score"], reverse=True)

def PE_recs(clinical_info: str, ddx: dict, retrieved_rows: list):
    evidence = build_evidence_block(retrieved_rows)
    dx_list = ddx.get("differential", [])

    prompt = f"""
Clinical scenario (no PHI):
{clinical_info}

Broad differential:
- {"; ".join(dx_list) if dx_list else "None provided"}

Retrieved evidence (likelihood ratios):
{evidence}

Instructions:
- Choose maneuvers that BEST ADDRESS the presenting problems and highest-risk differential given the scenario.
- Use retrieved LR evidence whenever it is relevant.
- If the retrieved evidence is missing an important domain suggested by the scenario/differential, you may include up to 2 additional high-yield maneuvers not in the evidence, but clearly label them as "scenario-driven (no LR provided)".
- Do NOT recommend maneuvers that only make sense for a diagnosis that is not supported by the presenting problems.

Task:
Return exactly 5 maneuvers, each as:
1) Maneuver / finding
2) How to perform (1–2 lines)
3) What + suggests / what - suggests (1 line)
4) LR+/LR- if available from retrieved evidence, otherwise write "LR: not available"
5) Which dx in the differential it helps adjudicate (comma-separated)
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert academic physician focused on medical education. Use the scenario to prioritize and the LR table to quantify when relevant."},
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
k = st.slider("How many evidence rows to retrieve", min_value=5, max_value=40, value=10)

if st.button("Recommend physical exam maneuvers"):
    if not clinical_info.strip():
        st.error("Please enter clinical information.")
    else:
        with st.spinner("Generating broad differential…"):
            ddx = generate_differential(clinical_info, n=8)

        st.subheader("Broad differential (LLM step 1)")
        st.write(ddx.get("differential", []))

        dx_queries = build_dx_queries(ddx, max_dx=15)

        with st.spinner("Retrieving evidence from LR table (local)…"):
            retrieved_all = retrieve_lr_rows_multi(dx_queries, k_each=4)
            retrieved = retrieved_all[:k]  # final k to show/pass to GPT

        st.subheader("Top retrieved LR-table rows (local)")
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

        with st.spinner("Generating learner-focused recommendations (LLM step 2)…"):
            recs = PE_recs(clinical_info, ddx, retrieved)

        st.subheader("Recommendations")
        st.write(recs)
