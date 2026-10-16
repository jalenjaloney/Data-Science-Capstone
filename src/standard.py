"""The standard purchase table every loader returns.

One row per purchase line, any vendor, any fiscal year:

    vendor       str     e.g. "Sysco"
    fiscal_year  int     e.g. 2026 (FY26 = Jul 2025 - Jun 2026)
    item         str     vendor's item/product code, if any
    desc         str     product description, upper-cased
    category     str     one of CATEGORIES (hand-assigned "Type" column)
    cert         str     certification codes as recorded, e.g. "V", "MSC, H"
    hall         str     dining location, if known
    usd          float   line dollars (negative for rebates)
    weight_lb    float   line weight in pounds, if known

Downstream code (the classifier, summaries) only ever sees this table, so
adding a vendor or year means writing one loader, nothing else.
"""
import pandas as pd

STANDARD_COLUMNS = [
    "vendor", "fiscal_year", "item", "desc", "category", "cert", "hall",
    "usd", "weight_lb",
]

# The FY26 workbook's category vocabulary. Loaders map vendor-specific
# spellings onto these.
CATEGORIES = {
    "Produce", "Dairy/Eggs", "Red Meat", "Poultry", "Seafood", "Pantry",
    "Bakery", "Beverage", "Frozen/Refrigerated", "Supplies/Non-Food",
    "Rebate/Allowance", "Unknown",
}

CATEGORY_ALIASES = {
    "Beverages": "Beverage",
    "Equipment": "Supplies/Non-Food",
    "Refrigerated/Frozen": "Frozen/Refrigerated",
    "Meat": "Red Meat",
}


def normalize_category(s: pd.Series) -> pd.Series:
    """Strip, apply aliases, and send blanks to "Unknown"."""
    s = s.astype("string").str.strip().replace(CATEGORY_ALIASES)
    return s.fillna("Unknown").replace("", "Unknown")


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Check a loader's output matches the standard; return it in order."""
    missing = set(STANDARD_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"missing standard columns: {sorted(missing)}")
    bad = set(df["category"].dropna()) - CATEGORIES
    if bad:
        raise ValueError(f"categories not in the standard vocabulary: {bad}")
    if df["usd"].isna().any():
        raise ValueError("usd has missing values")
    return df[STANDARD_COLUMNS]
