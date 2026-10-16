"""FY26 non-Sysco vendors: the per-vendor tabs of Food_2026_final.xlsx.

Most tabs share one template (Product, an amount column, Type,
Certification); amount headers vary ("Amount with tax", "Amount no tax",
"Price", ...). These tabs have no hall and no weight.

Used for now only to size Sysco's share of FY26 spend; the classifier will be
applied to these vendors in a later pass.
"""
import re
from pathlib import Path

import pandas as pd

from standard import normalize_category, validate

# Tabs that are summaries/charts, not purchase lines.
NON_VENDOR_TABS = {
    "All Vendors", "Final Statistics for Tableau", "Tableau- All Breakdown Pie",
    "Past Categories", "Tableau- Pie Charts", "Tableau- Packed Bubble Chart",
    "Final Weights", "Sysco",
}

VENDOR_NAMES = {
    "kikka": "Kikka",
    "nestle": "Nestle",
    "unfi": "UNFI",
    "Natures Produce": "Nature's Produce",
}


def _usd(tab: pd.DataFrame) -> pd.Series:
    """Line dollars: prefer a with-tax column, fall back to any other
    amount/price column row by row (Botrista fills only one per row)."""
    cols = [c for c in tab.columns if re.search(r"amount|price", c, re.I)]
    cols.sort(key=lambda c: 0 if re.search(r"w(ith)?\.? ?tax", c, re.I) else 1)
    usd = pd.to_numeric(tab[cols[0]], errors="coerce")
    for c in cols[1:]:
        usd = usd.fillna(pd.to_numeric(tab[c], errors="coerce"))
    return usd


def load(path: Path) -> pd.DataFrame:
    """Return every non-Sysco vendor tab as one standard table.

    Kept rows have a dollar amount and a product name; this drops blank rows
    and the hand-added TOTAL rows (Wismettac, Le Chef Bakery).
    """
    frames = []
    for sheet in pd.ExcelFile(path).sheet_names:
        if sheet in NON_VENDOR_TABS:
            continue
        tab = pd.read_excel(path, sheet_name=sheet)
        tab.columns = [str(c).strip() for c in tab.columns]
        if "Product" not in tab.columns:   # e.g. "NUMI Tea": "No purchases"
            continue
        usd = _usd(tab)
        desc = tab["Product"].astype("string").str.strip().str.upper()
        keep = usd.notna() & desc.notna() & (desc != "") & (desc != "TOTAL")
        frames.append(pd.DataFrame({
            "vendor": VENDOR_NAMES.get(sheet, sheet),
            "fiscal_year": 2026,
            "item": pd.NA,
            "desc": desc[keep],
            "category": normalize_category(tab.loc[keep, "Type"]),
            "cert": tab.loc[keep, "Certification"].astype("string").str.strip(),
            "hall": pd.NA,
            "usd": usd[keep],
            "weight_lb": float("nan"),
        }))
    return validate(pd.concat(frames, ignore_index=True))
