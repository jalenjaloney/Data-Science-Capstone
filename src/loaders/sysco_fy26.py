"""FY26 Sysco: the curated "Sysco" tab, reconciled against Sysco's raw export.

Two sources:
  * the "Sysco" tab of Food_2026_final.xlsx (Type, certification codes and
    Location added by Dining/Sustainability staff), and
  * the raw export Sysco_FY26.xlsx (one block per customer account, each
    ending in a "Totals For <acct#>" row).

We analyze the tab; the raw export is the ground truth we reconcile against.
"""
from pathlib import Path

import pandas as pd

from standard import normalize_category, validate

# The first 14 columns of the tab. Columns after these are a pivot-style
# certification summary, not row-level data.
TAB_COLS = [
    "Item #", "Pack", "Size", "Brand", "Description", "MPC Code", "CW", "Cases",
    "Weight", "Avg$/Lb", "Total Sales $", "Type", "Sustainability Criteria",
    "Location",
]

# Sysco customer account -> hall. Frary and Oldenborg each have two accounts.
ACCOUNT_HALL = {
    176830: "Coop Fountain",
    196448: "Cafe 47",
    607838: "Oldenborg",   # OLDENBORG CATERING
    608414: "Frary",       # FRARY APP REQ
    938613: "Frank",       # FRANK PRE APP
    938621: "Oldenborg",   # OLDENBORG PRE APP
    938639: "Frary",       # FRARY PRE APP
}

# Raw export column positions (0-indexed; the export has a 2-row header).
RAW_LABEL_COL = 1     # "Totals For <acct#>"
RAW_WEIGHT_COL = 13   # "Weight"
RAW_DOLLAR_COL = 15   # "Total Sales $"


def load(path: Path) -> pd.DataFrame:
    """Read the Sysco tab and return the standard table.

    Rows whose Total Sales $ is not numeric are dropped: section headers,
    spacers, and item lines with blank dollars and blank weight (ordered,
    never billed), so no total changes. Counts are in ``attrs['cleaning_log']``.
    """
    raw = pd.read_excel(path, sheet_name="Sysco", usecols=range(14))
    raw.columns = TAB_COLS
    usd = pd.to_numeric(raw["Total Sales $"], errors="coerce")
    tab = raw[usd.notna()]
    dropped = raw[usd.isna()]

    desc = tab["Description"].astype("string").str.strip().str.upper()
    df = pd.DataFrame({
        "vendor": "Sysco",
        "fiscal_year": 2026,
        "item": tab["Item #"].astype("string"),
        "desc": desc.replace("", pd.NA),
        "category": normalize_category(tab["Type"]),
        "cert": tab["Sustainability Criteria"].astype("string").str.strip(),
        "hall": tab["Location"].astype("string").str.strip(),
        "usd": usd[usd.notna()],
        "weight_lb": pd.to_numeric(tab["Weight"], errors="coerce"),
    }).reset_index(drop=True)

    df.attrs["cleaning_log"] = {
        "rows read": len(raw),
        "dropped: no numeric dollars": len(dropped),
        "  of which item lines": int(
            (dropped["Description"].notna() & dropped["Item #"].notna()).sum()),
        "rows kept": len(df),
        "kept: rebate/allowance": int((df["category"] == "Rebate/Allowance").sum()),
        "kept: no description": int(df["desc"].isna().sum()),
        "kept: no hall": int(df["hall"].isna().sum()),
    }
    return validate(df)


def load_raw_totals(path: Path) -> pd.DataFrame:
    """Return the 7 "Totals For <acct#>" rows of the raw export."""
    raw = pd.read_excel(path, header=None)
    label = raw[RAW_LABEL_COL].astype("string")
    tot = raw[label.str.startswith("Totals For", na=False)]
    out = pd.DataFrame({
        "account": label[tot.index].str.extract(r"(\d+)")[0].astype(int).values,
        "raw_usd": pd.to_numeric(tot[RAW_DOLLAR_COL]).values,
        "raw_weight_lb": pd.to_numeric(tot[RAW_WEIGHT_COL]).values,
    })
    out["hall"] = out["account"].map(ACCOUNT_HALL)
    return out


def reconcile(std: pd.DataFrame, raw_totals: pd.DataFrame) -> pd.DataFrame:
    """Per-hall dollars and pounds: standard table vs raw export.

    Lines with no hall are not in the raw export; they get their own row so
    the grand total still adds up.
    """
    w = (std.assign(hall=std["hall"].fillna("(no hall)"))
         .groupby("hall")
         .agg(lines=("usd", "size"), usd=("usd", "sum"),
              weight_lb=("weight_lb", "sum")))
    r = raw_totals.groupby("hall").agg(
        raw_accounts=("account", "size"), raw_usd=("raw_usd", "sum"),
        raw_weight_lb=("raw_weight_lb", "sum"))
    out = w.join(r, how="outer")
    out["usd_diff"] = (out["usd"] - out["raw_usd"]).round(2)
    out["weight_diff_lb"] = (out["weight_lb"] - out["raw_weight_lb"]).round(3)
    out.loc["TOTAL"] = out.sum(numeric_only=True)
    return out
