import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INDEX_PATH = "aipe_mcgee71_local.faiss"
META_PATH = "aipe_mcgee71_local_meta.json"

model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index(INDEX_PATH)
meta = json.load(open(META_PATH))

def search(query, k=5):
    qv = model.encode([query], normalize_embeddings=True)
    qv = np.array(qv, dtype="float32")
    scores, idxs = index.search(qv, k)
    for score, idx in zip(scores[0], idxs[0]):
        r = meta[idx]
        print(f"\nscore={score:.3f}  {r['Condition']} — {r['Finding']}")
        print(f"LR+={r.get('LR_plus')}  LR-={r.get('LR_minus')}  Pretest={r.get('Pretest_low')}-{r.get('Pretest_high')}")
        # print(r["text_for_embedding"])

search("appendicitis rebound tenderness", k=5)
