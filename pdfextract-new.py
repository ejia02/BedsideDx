import pdfplumber
import pandas as pd

#jama  rational clinical examination 

# === File paths ===
pdf_path = "AppendixChp71.pdf"
csv_out = "AppendixChp71_Table71_1.csv"
xlsx_out = "AppendixChp71_Table71_1.xlsx"

# === Page range (2–41) ===
start_page = 2   # as numbered in the PDF
end_page = 41

all_data = [] #empty list

with pdfplumber.open(pdf_path) as pdf:
    for page_num in range(start_page - 1, end_page):  # zero-indexed, range is iterating by 1 from start=pg - 1 to end-page
        page = pdf.pages[page_num]
        tables = page.extract_tables()
        if not tables:
            continue

        for table in tables:
            df = pd.DataFrame(table)
            # Drop fully empty rows
            df = df.dropna(how="all")
            df = df.loc[(df.astype(str).apply(lambda x: "".join(x), axis=1).str.strip() != "")]
            if df.empty:
                continue
            all_data.append(df)

# === Combine all fragments ===
if all_data:
    combined = pd.concat(all_data, ignore_index=True)
    # Clean up column names and strip whitespace
    combined = combined.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Optional: rename headers to your desired format if the first row is header
    headers = ["Finding", "Positive LR (95% CI)", "Negative LR (95% CI)", "Pretest Probability (Range)"]
    if len(combined.columns) >= 4:
        combined.columns = headers[:len(combined.columns)]

    # === Save ===
    combined.to_csv(csv_out, index=False)
    combined.to_excel(xlsx_out, index=False)

    print(f"✅ Combined table saved as:\n  • {csv_out}\n  • {xlsx_out}")
else:
    print("⚠️ No tables detected on pages 2–41.")
