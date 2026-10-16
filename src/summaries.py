"""Summary tables built on the standard (and classified) purchase table."""
import pandas as pd

PROTEIN_CATEGORIES = ["Red Meat", "Poultry", "Seafood", "Dairy/Eggs"]

# Nested protein definitions: (animal classes, plant classes). Each adds a
# matched animal/plant pair to the one before; C is the reference.
DEFINITIONS = {
    "A": (["flesh"], ["analog", "legume_soy"]),
    "B": (["flesh", "egg"], ["analog", "legume_soy", "nut_seed"]),
    "C": (["flesh", "egg", "dairy"],
          ["analog", "legume_soy", "nut_seed", "plant_dairy_alt"]),
    "D": (["flesh", "egg", "dairy", "mixed_animal"],
          ["analog", "legume_soy", "nut_seed", "plant_dairy_alt",
           "mixed_plant"]),
}


def plant_share(df: pd.DataFrame) -> pd.DataFrame:
    """Plant share of protein under each definition, by dollars and pounds."""
    rows = {}
    for name, (animal, plant) in DEFINITIONS.items():
        a = df[df["protein_class"].isin(animal)]
        p = df[df["protein_class"].isin(plant)]
        rows[name] = {
            "plant_usd": p["usd"].sum(),
            "animal_usd": a["usd"].sum(),
            "plant_share_usd": p["usd"].sum()
            / (p["usd"].sum() + a["usd"].sum()),
            "plant_share_lb": p["weight_lb"].sum()
            / (p["weight_lb"].sum() + a["weight_lb"].sum()),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


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
