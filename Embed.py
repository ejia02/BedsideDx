import pandas as pd

clean_data = "AppendixChp71_Table71_Clean.csv"
df = pd.read_csv(clean_data)

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
        parts.append(
            f"Pretest probability range: {int(row['Pretest_low'])}-{int(row['Pretest_high'])}%"
        )
    return " | ".join(parts)

df["text_for_embedding"] = df.apply(build_embedding_text, axis=1)

# Stable ID
df["doc_id"] = [f"mcgee71_{i}" for i in range(len(df))]


