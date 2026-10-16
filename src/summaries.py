"""Summary tables built on the standard (and classified) purchase table."""
import pandas as pd

PROTEIN_CATEGORIES = ["Red Meat", "Poultry", "Seafood", "Dairy/Eggs"]


def vendor_scope(std: pd.DataFrame) -> pd.DataFrame:
    """Spend per vendor, its share of all spend, and spend in the
    protein-coded categories. Sorted by total spend."""
    total = std.groupby("vendor")["usd"].sum().rename("usd")
    by_cat = (std[std["category"].isin(PROTEIN_CATEGORIES)]
              .pivot_table(index="vendor", columns="category", values="usd",
                           aggfunc="sum", fill_value=0)
              .reindex(columns=PROTEIN_CATEGORIES, fill_value=0))
    out = pd.concat([total, by_cat], axis=1).fillna(0)
    out.insert(1, "share", out["usd"] / out["usd"].sum())
    return out.sort_values("usd", ascending=False)


def class_crosstab(df: pd.DataFrame, value: str = "usd") -> pd.DataFrame:
    """Category x protein_class table of ``value`` with row/column totals."""
    return pd.crosstab(df["category"], df["protein_class"], values=df[value],
                       aggfunc="sum", margins=True, margins_name="Total"
                       ).fillna(0)


def items(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Distinct descriptions under ``mask`` with dollars, pounds and rule."""
    return (df[mask]
            .groupby(["desc", "category", "protein_class", "rule"],
                     dropna=False)
            .agg(usd=("usd", "sum"), weight_lb=("weight_lb", "sum"))
            .reset_index()
            .sort_values("usd", ascending=False))
