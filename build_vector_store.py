import json
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

INPUT_CSV = "AppendixChp71_Table71_Clean.csv"
OUT_INDEX = "aipe_mcgee71_local.faiss"
OUT_META = "aipe_mcgee71_local_meta.json"

# 1) Load
df = pd.read_csv(INPUT_CSV)

# 2) Your embedding text builder (keep your current function)
def build_embedding_text(row):
    parts = []
    if pd.notna(row.get("Condition")):
        parts.append(f"Condition: {row['Condition']}")
    if pd.notna(row.get("Finding")):
        parts.append(f"Finding / Maneuver: {row['Finding']}")
    if pd.notna(row.get("LR_plus")):
        parts.append(f"Positive likelihood ratio: {row['LR_plus']}")
    if pd.notna(row.get("LR_minus")):
        parts.append(f"Negative likelihood ratio: {row['LR_minus']}")
    if pd.notna(row.get("Pretest_low")) and pd.notna(row.get("Pretest_high")):
        parts.append(f"Pretest probability range: {int(row['Pretest_low'])}-{int(row['Pretest_high'])}%")
    return " | ".join(parts)

df["text_for_embedding"] = df.apply(build_embedding_text, axis=1)
df["doc_id"] = [f"mcgee71_{i}" for i in range(len(df))]

texts = df["text_for_embedding"].fillna("").tolist()

# 3) Local embedding model (no API)
# Good default: small, fast, high-quality for semantic search
model = SentenceTransformer("all-MiniLM-L6-v2")

emb = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True,   # important for cosine similarity
)

emb = np.array(emb, dtype="float32")

# 4) Build FAISS index (cosine via inner product, since vectors normalized)
dim = emb.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(emb)

faiss.write_index(index, OUT_INDEX)

# 5) Save metadata for retrieval/display
meta_cols = ["doc_id", "Condition", "Finding", "LR_plus", "LR_minus", "Pretest_low", "Pretest_high", "text_for_embedding"]
meta = df[meta_cols].to_dict(orient="records")

with open(OUT_META, "w") as f:
    json.dump(meta, f, indent=2)

print(f"Saved FAISS index: {OUT_INDEX}")
print(f"Saved metadata:   {OUT_META}")
print(f"Rows indexed:     {len(df)}")
print(f"Embedding dim:    {dim}")
