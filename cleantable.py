import pandas as pd
import re 

raw_data = "AppendixChp71_Table71_1.csv"
clean_data = "AppendixChp71_Table71_Clean.csv"

#returns a clean string from CSV table
def normalize_text(x: str) -> str:
    if pd.isna(x): #if value is missing, replace with an empty string
        return ""
    x = str(x) #convert the value to a string
    x = x.replace("\xa0", " ") # Replace non-breaking spaces with normal spaces
    x = x.replace("\u2013", "-").replace("\u2014", "-") #replace special dashes w/ normal hypen
    x = x.replace("…", "") #remove ..., replace with empty string 
    x = x.replace("\u201c", '"').replace("\u201d", '"') #replace curly smart quotes with straight quotes (using unicode)
    x = re.sub(r"\s+", " ", x).strip() #Use a regular expression to collapse multiple spaces/newlines/tabs into one space. Example: "Abnormal\n   clock   test" → "Abnormal clock test"
    if len(x) >= 2 and x[0] == '"' and x[-1] == '"': #if string is surrounded by quotes remove outer quotes and trim any lefover spaces
        x = x[1:-1].strip()
    return x 

def parse_lr_cell(s: str):
    """
    Accepts things like:
      '4.6 (3.1, 6.8)'
      '0.2 (0.1, 0.5)'
      '...' or ''  -> returns (None, None, None)
    Returns (value, ci_low, ci_high) as floats or None.
    """
    s = normalize_text(s) #removes line breaks, extra spaces, culry quotes etc. 
    if not s or s in {".", "..."}: #if the cell is empty (not s) or has just dots--> function returns three none values - placeholders for (value, ci low, ci high)
        return None, None, None
    m = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*\(\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\)\s*$", s) #regex to capture 3 numbers
    if m: #if the regex expression matched, extract the three numbers, convert it from string to float, return as a tuple (v, low, high)
        v, lo, hi = map(float, m.groups()) #m.groups returns a tuple of strings, one string for each captured value, map()--> applies another funciton to every element of an iterable (in this case m.groups()
        ## apply float function to every element of m.groups()--> does float(val) for val in m.groups()
        return v, lo, hi
    # Sometimes you only have the point value without CI
    m2 = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*$", s) #regex if only have one number (ex no low or hi ci)
    if m2:
        return float(m2.group(1)), None, None
    # Could be '0 (0, 0.7)' or '0.0 (0, 0.7)' → handled by main regex
    return None, None, None

def parse_pretest(s: str):
    """Parse things like '22–73' or '46' into (low, high) integers."""
    if not s:
        return None, None

    s = normalize_text(s)
    # normalize en dash to hyphen if any remain
    s = s.replace("–", "-")

    # range like 22-73
    m = re.match(r"^\s*([0-9]+)\s*-\s*([0-9]+)\s*$", s)
    if m:
        return int(m.group(1)), int(m.group(2))

    # single value like 46
    m2 = re.match(r"^\s*([0-9]+)\s*$", s)
    if m2:
        v = int(m2.group(1))
        return v, v

    return None, None

def clean_condition_name(s: str) -> str | None:
    """
    Turn things like:
      'EBM Box 6.1 Dementia and delirium' -> 'Dementia and delirium'
      'Chapter 7 Stance and gait'         -> 'Stance and gait'
      'Diagnosing appendicitis'           -> 'Appendicitis'
      'Detecting meningitis'              -> 'Meningitis'
      'Predicting falls'                  -> 'Falls'
    """
    if not isinstance(s, str):
        return None
    s = s.strip()

    # Remove EBM Box <number>
    s = re.sub(r"EBM\s*Box\s*\d+(\.\d+)?\s*", "", s, flags=re.IGNORECASE)

    # Remove Chapter <number>
    s = re.sub(r"Chapter\s*\d+\s*", "", s, flags=re.IGNORECASE)

    # Remove leading verbs (Diagnosing/Detecting/Predicting)
    s = re.sub(r"^(Diagnosing|Detecting|Predicting)\s+", "", s, flags=re.IGNORECASE)

    s = s.strip()
    return s or None

def main():
    # 1. Read raw CSV
    df = pd.read_csv(raw_data)

    # 2. Normalize every string cell in the DataFrame
    # (applymap is deprecated in new pandas; map works elementwise)
    df = df.map(lambda v: normalize_text(v) if isinstance(v, str) else v)

    # 3. Drop duplicate header row inside the data (where Finding literally == "Finding")
    df = df[df["Finding"] != "Finding"]

    # 4. Identify header rows:
    #    Finding has text, other three columns are empty/NaN
    header_mask = (
        df["Finding"].notna() & df["Finding"].astype(str).str.strip().ne("") &
        df["Positive LR (95% CI)"].isna() &
        df["Negative LR (95% CI)"].isna() &
        df["Pretest Probability (Range)"].isna()
    )

    # 5. Create raw condition labels only on header rows
    df["Condition_raw"] = None
    df.loc[header_mask, "Condition_raw"] = (
        df.loc[header_mask, "Finding"].apply(clean_condition_name)
    )

    # 6. Forward-fill so each row inherits the last condition
    df["Condition"] = df["Condition_raw"].ffill()

    # 7. Drop the header rows themselves
    df = df[~header_mask]
    df = df.drop(columns=["Condition_raw"])

    # 8. Parse LR+ and LR– into numeric columns
    df[["LR_plus", "LR_plus_CI_low", "LR_plus_CI_high"]] = df[
        "Positive LR (95% CI)"
    ].apply(lambda s: pd.Series(parse_lr_cell(s)))

    df[["LR_minus", "LR_minus_CI_low", "LR_minus_CI_high"]] = df[
        "Negative LR (95% CI)"
    ].apply(lambda s: pd.Series(parse_lr_cell(s)))

    # 9. Parse pretest probability range
    df[["Pretest_low", "Pretest_high"]] = df[
        "Pretest Probability (Range)"
    ].apply(lambda s: pd.Series(parse_pretest(s)))

    # 10. Optionally drop the original messy text columns
    df = df.drop(
        columns=[
            "Positive LR (95% CI)",
            "Negative LR (95% CI)",
            "Pretest Probability (Range)",
        ]
    )

    # 11. Quick preview in the terminal
    print(df.head(20).to_string())

    # 12. Save cleaned CSV
    df.to_csv(clean_data, index=False)
    print(f"\nSaved cleaned data to {clean_data}")


if __name__ == "__main__":
    main()
