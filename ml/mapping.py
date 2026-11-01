"""Map NYC 311 housing complaints to the project's categories and urgency.

NOTE: these are WEAK LABELS. 311 has no urgency field, and several complaint
types do not fit our categories exactly. Every choice here is recorded in the
data card. Complaint types not listed are dropped (not housing maintenance).
"""

# complaint type (uppercase) -> category
COMPLAINT_TO_CATEGORY = {
    "PLUMBING": "plumbing",
    "WATER LEAK": "plumbing",
    "WATER SYSTEM": "plumbing",
    "ELECTRIC": "electricity",
    # HEAT/HOT WATER is split by the additional_details field (see to_category)
    "HEAT/HOT WATER": "ac",
    "HEATING": "ac",
    "DOOR/WINDOW": "carpentry",
    "FLOORING/STAIRS": "carpentry",
    "RODENT": "pest control",
    "PAINT/PLASTER": "general",
    "PAINT - PLASTER": "general",
    "GENERAL CONSTRUCTION": "general",
    "GENERAL": "general",
    "APPLIANCE": "general",
    "SAFETY": "general",
    "ELEVATOR": "general",
    "OUTSIDE BUILDING": "general",
    "NONCONST": "general",
    "UNSANITARY CONDITION": "general",  # overridden to pest control when descriptor says pests
}

# descriptor keywords that override the category
DESCRIPTOR_OVERRIDES = [
    (("PEST", "ROACH", "MICE", "MOUSE", "RAT", "BED BUG", "BEDBUG", "VERMIN"), "pest control"),
]

HIGH_KEYWORDS = ("LEAK", "FLOOD", "BURST", "SEWAGE", "SEWER", "SPARK", "FIRE", "SMOKE",
                 "NO GAS", "NO ELECTRICITY", "NO POWER", "NO HEAT", "NO WATER",
                 "ENTIRE BUILDING", "COLLAPS", "EXPOSED WIRE", "HOLE IN CEILING")
LOW_KEYWORDS = ("PAINT", "PLASTER", "PEELING", "CHIPPED", "FLAKING", "COSMETIC", "CRACK",
                "SCRATCH", "CABINET", "MINOR")


def to_category(complaint_type: str, descriptor: str, details: str = ""):
    c = (complaint_type or "").strip().upper()
    d = (descriptor or "").strip().upper()
    x = (details or "").strip().upper()
    if c not in COMPLAINT_TO_CATEGORY:
        return None
    # HEAT/HOT WATER: use the detail field to separate heating from hot water
    if c in ("HEAT/HOT WATER", "HEATING"):
        has_hw, has_heat = "HOT WATER" in x, "HEAT" in x.replace("HOT WATER", "")
        if has_hw and not has_heat:
            return "plumbing"
        if has_heat and not has_hw:
            return "ac"
        return None  # ambiguous or empty: drop rather than guess
    for words, cat in DESCRIPTOR_OVERRIDES:
        if any(w in f"{c} {d} {x}" for w in words):
            return cat
    return COMPLAINT_TO_CATEGORY[c]


def to_urgency(complaint_type: str, descriptor: str, details: str = "") -> str:
    s = f"{complaint_type} {descriptor} {details}".upper()
    if any(k in s for k in HIGH_KEYWORDS):
        return "high"
    if any(k in s for k in LOW_KEYWORDS):
        return "low"
    return "medium"