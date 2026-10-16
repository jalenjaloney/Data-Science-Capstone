"""Ordered regex classifier: (description, category) -> protein class.

Rules are tried in order; the first match wins. Every row records the name of
the rule that fired, so any classification can be audited and any rule's
effect can be listed.

Inputs are only a description and a standard category (see standard.py), so
the same rules apply to any vendor or year. Vendor abbreviations are handled
inside the patterns (CHS/CHSE/CHESE = cheese, BF = beef, SAUS/SSG = sausage).

Rule order (each block can only claim rows earlier blocks left alone):
  1. non-protein: whole categories, blank descriptions, leading words
     (SAUCE, SOUP, BUN, ...), butter/cream/creamer, known false hits
  2. plant dairy alternatives (oat/soy/almond/coconut milk, vegan cheese)
  3. plant analogs (Beyond, Impossible, "vegan chicken", falafel). These come
     BEFORE animal words because e.g. "CHICKEN CUTLET BRD VEGAN" is filed
     under Poultry
  4. mixed dishes (dumplings, ravioli, breakfast sandwiches, pizza): plant if
     they name a plant filling and no animal word, otherwise animal
  5. flesh: poultry, seafood, beef/lamb, pork (in that order, so
     "BACON TURKEY" is poultry and "SHRIMP BACON WRAPPED" is seafood)
  6. egg, 7. dairy, 8. legumes/soy, 9. nuts/seeds
  10. fallback: unmatched Red Meat/Poultry/Seafood rows count as flesh;
      everything else is non-protein
"""
import re
from dataclasses import dataclass

import pandas as pd

ANIMAL_CLASSES = ["flesh", "egg", "dairy", "mixed_animal"]
PLANT_CLASSES = ["analog", "legume_soy", "nut_seed", "plant_dairy_alt",
                 "mixed_plant"]
CLASS_GROUP = {
    **{c: "animal" for c in ANIMAL_CLASSES},
    **{c: "plant" for c in PLANT_CLASSES},
    "non_protein": "non_protein",
}

NON_PROTEIN_CATEGORIES = {"Supplies/Non-Food", "Rebate/Allowance"}
FLESH_CATEGORIES = {"Red Meat": "red_meat_unspecified", "Poultry": "poultry",
                    "Seafood": "seafood"}


def _rx(*parts: str) -> re.Pattern:
    return re.compile("|".join(parts))


# --- word lists reused by several rules -----------------------------------

POULTRY = r"\b(CHICKEN|CHKN|CHIX|TURKEY|DUCK|POULET|QUAIL)\b"
SEAFOOD = (r"\b(SALMON|SHRIMP|LOBSTER|LBSTR|CRAB|COD|TUNA|AHI|SNAPPER|"
           r"TILAPIA|TROUT|HALIBUT|MUSSELS?|CALAMARI|SQUID|SCALLOPS?|SCLP|"
           r"CLAMS?|POLLOCK|CATFISH|PANGASIUS|ANCHOVY|FISH|OYSTERS?|EEL|"
           r"UNAGI|SASHIMI|NIGIRI|SUSHI|YELLOWTAIL|ALBACORE|BASS|MAHI|"
           r"SARDINES?|OCTOPUS|PRAWNS?)\b")
BEEF_LAMB = (r"\b(BEEF|BF|STEAK|STK|LAMB|VEAL|OXTAIL|BRISKET|RIBEYE|RIB EYE|"
             r"TRI TIP|SIRLOIN|SIRL|TENDERLOIN|STRIPLOIN|STRIP LN|CHUCK|"
             r"SHORT RIB|SKIRT|FLANK|HANGER|FLAT IRON|BIRRIA|BARBACOA|"
             r"CARNE ASADA|PASTRAMI|BULGOGI|BULGOLGI|TRIPE|TONGUE|WAGYU|"
             r"MEATBALL|FRANKS?|PRIME RIB|CORNED)\b")
# PK alone usually means "pack" in Sysco descriptions; only PK/BF is pork.
PORK = (r"\b(PORK|PK/BF|BACON|BAC|BCN|HAM|PROSCIUTTO|PANCETTA|CAPICOLA|"
        r"GUANCIALE|SALAMI|PEPPERONI|MORTADELLA|SOPRESSATA|CHORIZO|"
        r"SAUSAGE|SAUS|SSG|BRAT|BRATWURST|ANDOUILLE|LINGUICA|CARNITAS|"
        r"SPAM|CHAR SIU|JAMON)\b")
EGG = r"\bEGGS?\b"
CHEESE = (r"\b(CHEESE|CHS|CHSE|CHESE|CSE|MOZZ|MOZZARELLA|PARM|PARMESAN|"
          r"CHED|CHDR|CHEDDAR|CHEDDR|PANEER|RICOTTA|QUESO|COTIJA|HALLOUMI|"
          r"BURRATA|FETA|BRIE|GOUDA|MASCARPONE|OAXACA|PROVOLONE|PROV|"
          r"SWISS|SWIS|GRUYERE|HAVARTI|MANCHEGO|GORGONZOLA|BOURSIN|"
          r"CHIHUAHUA)\b")
MILK = r"\b(MILK|BUTTERMILK|KEFIR)\b"
YOGURT = r"\b(YOGURT|YOGHURT)\b"
ANIMAL_WORD = _rx(POULTRY, SEAFOOD, BEEF_LAMB, PORK, EGG, CHEESE, MILK,
                  r"\bCH\b", r"\bMEAT\b")

MIXED_DISH = (r"DUMPLING|POTSTICKER|PTSTKR|GYOZA|SHUMAI|HAR GOW|WONTON|"
              r"MANDU|\bBAO\b|EGG ?ROLL|SPRING ?ROLL|EMPANADA|TAMALE|"
              r"TAQUITO|BURRITO|\bPUP\b|PUPUSA|RAVIOLI|TORTEL|GNOCCHI|"
              r"\bMAC(ARONI)?\b|^PIZZA|CORN DOG|SANDWICH|"
              r"BISCUIT|CROISSANT|BAGEL|CRAB CAKE|GRITS|SAMOSA|"
              r"LASAGNA|ENCHILADA|CALZONE|POT PIE|PHYLLO|CHILI .* W/? ?BEAN|"
              r"^BREAKFAST\b|^PASTRY\b|ENCROU|EN CROUTE|WELLINGTON")
PLANT_FILLING = (r"\bVEG\w*|VEGAN|MUSHROOM|MSHRM|EDAMAME|SAMOSA|TOFU|"
                 r"PEANUT|\bBEANS?\b")

# Analog = a plant marker together with a meat-type noun (so "PRETZEL BITE
# SOFT VEGAN" is not an analog), or a brand/product that is always an analog.
ANALOG_BRAND = (r"BEYOND|IMPOSSIBLE|INCOGMEATO|GARDEIN|MORNINGSTAR|QUORN|"
                r"SOYRIZO|CHORIZO SOY|FALAFEL|SEITAN|TEMPEH|JACKFRUIT|"
                r"FRUIT JACK|BACON BIT IMITATION")
PLANT_BASED = r"PLANT ?BASED|PLANT BASE\b|PLNT-?BSD|\bP/B\b"
ANALOG_MARKER = r"\bVEGAN\b|VEGETARIAN|\bVGTRN\b|\bVEG\b|" + PLANT_BASED
MEAT_NOUN = (r"CHICKEN|CHKN|\bNUG|BURGER|PATTY|\bPTY\b|SAUSAGE|BRAT|"
             r"PEPPERONI|MEATBALL|CUTLET|TENDER|FILET|STEAK|TIPS|CHUNKS|"
             r"SHRED|CHORIZO|BACON|CRUMBLE|GROUND|TURKEY|BEEF|PORK|FISH|"
             r"TUNA|SHRIMP")

# Non-protein words at the start of a description. Anchoring keeps
# "CHICKEN NUGGET W/O SAUCE" as protein while "SAUCE STEAK" is not.
NP_LEADING = (
    r"^(SAUCE|SOUP|BASE|STOCK|GLACE|SEASONING|SPICE|MIX|BREADER|DRESSING|"
    r"OIL|CRACKER|BUN|BREAD|ROLL|PIE|CANDY|SNACK|BAR|BREAKFAST BAR|CEREAL|"
    r"COOKIE|DOUGH|CHEESECAKE|POPCORN|CROUTON|CHIP|POTATO|NOODLE|JUICE|"
    r"WATER|COFFEE|EXTRACT|PASTE|FLOUR|OAT|COCONUT|DRINK|PAN COATING|PLATE|"
    r"LETTUCE|GREEN|CABBAGE|GRAPE|RAISIN|JAM|BATTER|SQUASH|BANANA|PLANTAIN|"
    r"ICE CREAM|DONUT|CUPCAKE|MARGARINE|MAYO|MAYONNAISE|CHESTNUT|WAFFLE|"
    r"DANISH|CHOCOLATE|PRETZEL|VEG BLEND|COUSCOUS|RICE|GRAIN|SPROUT|"
    r"BUTTERSCOTCH|SUGAR|SYRUP|HONEY|VINEGAR|KETCHUP|MUSTARD|SALSA|PICKLE|"
    r"OLIVE|TEA|CAKE|BROWNIE|MUFFIN|EGGPLANT|ENTREE EGGPLANT|SEED SESAME|"
    r"CANNOLI)\b",
    r"^DIP\b(?! HUMMUS)",
    r"^TOPPING\b(?! PIZZA)",
    r"^PEAS?\b(?! BLACK ?EYE)",
    r"^PASTA\b(?!.*(RAVIOLI|TORTEL|GNOCCHI))",
    r"^BISCUIT\b(?!.*(SAUS|EGG|HAM|BAC))",
    r"^TORTILLA\b(?!.*(BEAN|CSE|CHEESE))",
    r"^SPREAD (COOKIE|CHOC|BUTTER)",
)
# "BUTTER" but not peanut/nut/sunflower butter or buttermilk.
NP_BUTTER = r"(?<!PEANUT )(?<!NUT )(?<!SUNFLOWER )(?<!SEED )\bBUTTER\b"
NP_CREAM = r"\b(CREAM|CREAMER|CREMA)\b|CHEESECAKE|MARGARINE"
NP_FALSE_HIT = (
    r"EGGPLANT|TEXAS TOAST|NOODLE EGG|EGG NOODLE",
    r"BEAN SPROUT|SPROUT BEAN|BEAN GREEN|GREEN BEAN|BEANS? \((BLUELAKE|"
    r"FRENCH|HARICOT|YELLOW WAX|ITALIAN GREEN)",
    r"COFFEE|ESPRESSO|ESPRSO|BEAN VANILLA|VANILLA BEAN|\bVAN BEAN\b|"
    r"BEAN VAN\b|BEAN THREAD|JELLY BEAN",
    r"FRY STEAK HOUSE|TEST STRIP|NUTELLA|SEEDLESS|NO SEED|POPPY ?SEED|"
    r"\bNORI\b",
    r"\b(CUMIN|CARAWAY|CORIANDER|FENNEL|CELERY|SESAME|MILLET) SEED",
)


@dataclass(frozen=True)
class Rule:
    name: str
    protein_class: str
    subtype: str
    pattern: re.Pattern | None = None   # must match the description
    require: re.Pattern | None = None   # must also match
    forbid: re.Pattern | None = None    # must NOT match
    categories: frozenset | None = None  # restrict to these categories

    def matches(self, desc: str, category: str) -> bool:
        if self.categories is not None and category not in self.categories:
            return False
        if self.pattern is not None and not self.pattern.search(desc):
            return False
        if self.require is not None and not self.require.search(desc):
            return False
        if self.forbid is not None and self.forbid.search(desc):
            return False
        return True


NP = "non_protein"
RULES = [
    # 1. non-protein
    Rule("np_category", NP, "", categories=frozenset(NON_PROTEIN_CATEGORIES)),
    Rule("np_leading_word", NP, "", _rx(*NP_LEADING)),
    Rule("np_butter", NP, "", _rx(NP_BUTTER), forbid=_rx(CHEESE, EGG, PORK)),
    Rule("np_cream", NP, "", _rx(NP_CREAM)),
    Rule("np_false_hit", NP, "", _rx(*NP_FALSE_HIT)),
    # Canned coconut milk (Chaokoh 98 FOZ, Roland, Thai Kitchen) is a cooking
    # ingredient that substitutes for cream in curries and desserts, not for
    # milk, so it is non-protein like dairy cream. Only beverage cartons
    # (Silk, Pacific, Simply), which Sysco describes as BARISTA or PURE,
    # count as a plant milk (rule plant_milk below).
    Rule("np_coconut_cooking", NP, "",
         _rx(r"\bMILK COCONUT\b", r"\bCOCONUT MILK\b"),
         forbid=_rx(r"BARISTA", r"\bPURE\b", r"BEVERAGE", r"\bBEV\b")),
    # 2. plant dairy alternatives
    Rule("plant_milk", "plant_dairy_alt", "milk_alt",
         _rx(r"\bMILK (OAT|SOY|ALMOND|COCONUT|RICE|CASHEW|HEMP|PEA)\b",
             r"\b(OAT|SOY|ALMOND|COCONUT|RICE|CASHEW) ?MILK\b",
             r"\bMILK \((ALMOND|OAT|SOY|COCONUT)\)")),
    Rule("plant_cheese_yogurt", "plant_dairy_alt", "cheese_yogurt_alt",
         _rx(CHEESE, YOGURT),
         require=_rx(r"\bVEGAN\b", r"DAIRY ?FRE", r"DAIRYFREE", r"NON-?DAIRY",
                     PLANT_BASED)),
    Rule("plant_egg", "plant_dairy_alt", "egg_alt", _rx(EGG),
         require=_rx(PLANT_BASED, r"JUST EGG", r"\bVEGAN\b")),
    # 3. plant analogs
    Rule("analog_brand", "analog", "analog", _rx(ANALOG_BRAND)),
    Rule("analog_marker", "analog", "analog", _rx(ANALOG_MARKER),
         require=_rx(MEAT_NOUN)),
    # 4. mixed dishes
    # Frittatas are mostly egg, so they count as egg even when the description
    # lists cheese or bacon (which would otherwise make them mixed or pork).
    Rule("egg_frittata", "egg", "egg", _rx(r"FRITTATA")),
    # Domain knowledge, not the description: chile rellenos are cheese-stuffed
    # peppers, but Sysco's descriptions ("CHILE RELLENO", "RELLENO CHILE")
    # never mention the cheese, so no animal word would catch them.
    Rule("mixed_relleno", "mixed_animal", "mixed_animal", _rx(r"RELLENO")),
    # Whole pizzas ("16 IN", 9/case) are mixed dishes via ^PIZZA in
    # MIXED_DISH; shredded pizza cheese is described "CHEESE PIZZA ...".
    Rule("mixed_plant", "mixed_plant", "mixed_plant", _rx(MIXED_DISH),
         require=_rx(PLANT_FILLING), forbid=ANIMAL_WORD),
    Rule("mixed_animal", "mixed_animal", "mixed_animal", _rx(MIXED_DISH),
         require=ANIMAL_WORD),
    # 5. flesh
    Rule("flesh_poultry", "flesh", "poultry", _rx(POULTRY)),
    Rule("flesh_seafood", "flesh", "seafood", _rx(SEAFOOD)),
    Rule("flesh_beef_lamb", "flesh", "beef_lamb", _rx(BEEF_LAMB)),
    Rule("flesh_pork", "flesh", "pork", _rx(PORK)),
    # 6. egg
    Rule("egg", "egg", "egg", _rx(EGG)),
    # 7. dairy
    Rule("dairy_cheese", "dairy", "cheese", _rx(CHEESE)),
    Rule("dairy_yogurt", "dairy", "yogurt", _rx(YOGURT)),
    Rule("dairy_milk", "dairy", "milk", _rx(MILK)),
    # 8. legumes / soy
    Rule("legume_soy", "legume_soy", "legume_soy",
         _rx(r"\bTOFU\b", r"HUMMUS", r"LENTIL", r"CHICKPEA", r"GARBANZO",
             r"EDAMAME", r"SOYBEAN", r"\bBEANS?\b", r"BLACK ?EYE")),
    # 9. nuts / seeds
    Rule("nut_seed", "nut_seed", "nut_seed",
         _rx(r"\bNUTS?\b", r"PEANUT", r"PNUT", r"ALMONDS?", r"CASHEW",
             r"WALNUTS?", r"PECANS?", r"PISTACH", r"MACADAMIA", r"HAZELNUT",
             r"HZLNT", r"FILBERT", r"PUMPKIN SEED", r"PEPITAS", r"SUNFLOWER",
             r"SUNBUTTER", r"\bCHIA\b", r"\bFLAX", r"HEMP SEED")),
]
FALLBACK_RULE = "fallback_category"
DEFAULT_RULE = "default_non_protein"
NO_DESC_RULE = "np_no_description"


def classify(desc, category) -> tuple[str, str, str]:
    """Return (protein_class, subtype, rule_name) for one purchase line."""
    category = "" if pd.isna(category) else str(category)
    if pd.isna(desc) or not str(desc).strip():
        return NP, "", NO_DESC_RULE
    desc = " ".join(str(desc).upper().split())
    for rule in RULES:
        if rule.matches(desc, category):
            return rule.protein_class, rule.subtype, rule.name
    if category in FLESH_CATEGORIES:
        return "flesh", FLESH_CATEGORIES[category], FALLBACK_RULE
    return NP, "", DEFAULT_RULE


def classify_frame(df: pd.DataFrame, desc: str = "desc",
                   category: str = "category") -> pd.DataFrame:
    """Add protein_class, protein_group, subtype and rule columns.

    Classifies each distinct (desc, category) pair once, then joins back.
    """
    pairs = df[[desc, category]].drop_duplicates()
    out = pd.DataFrame(
        [classify(d, c) for d, c in pairs.itertuples(index=False)],
        columns=["protein_class", "subtype", "rule"], index=pairs.index)
    pairs = pd.concat([pairs, out], axis=1)
    res = df.merge(pairs, on=[desc, category], how="left")
    res["protein_group"] = res["protein_class"].map(CLASS_GROUP)
    return res
