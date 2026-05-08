"""Make in India deterministic lookup helpers."""

# Local Content Requirements per PPO 2017 (as amended)
# Class I Local Supplier: >= 50% local content
# Class II Local Supplier: >= 20% but < 50% local content
# Non-Local Supplier: < 20% local content

LOCAL_CONTENT_THRESHOLDS = {
    "class_i": {
        "min_local_content": 50,
        "label": "Class I Local Supplier",
        "definition": "A supplier whose goods/services have local content >= 50%.",
    },
    "class_ii": {
        "min_local_content": 20,
        "label": "Class II Local Supplier",
        "definition": "A supplier whose goods/services have local content >= 20% but < 50%.",
    },
    "non_local": {
        "min_local_content": 0,
        "label": "Non-Local Supplier",
        "definition": "A supplier whose goods/services have local content < 20%.",
    },
}

# Eligibility Rules by Procurement Value
ELIGIBILITY_BY_VALUE = {
    "upto_50_lakh": {
        "range": "Up to Rs. 50 lakhs",
        "eligible": ["Class I", "Class II"],
        "purchase_preference": True,
        "note": (
            "For categories where the nodal ministry has confirmed sufficient local capacity and competition, "
            "only eligible local suppliers should ordinarily be considered in this value band."
        ),
    },
    "above_50_lakh_upto_200_cr": {
        "range": "Above Rs. 50 lakhs to Rs. 200 crore",
        "eligible": ["Class I", "Class II"],
        "purchase_preference": True,
        "note": (
            "Above Rs. 50 lakhs, the evaluation moves to the next purchase-preference rules. "
            "Class I and Class II local suppliers remain the default eligible classes under this lookup, "
            "and purchase preference for Class I over other eligible suppliers should be examined."
        ),
    },
    "above_200_cr": {
        "range": "Above Rs. 200 crore",
        "eligible": ["Class I", "Class II"],
        "purchase_preference": True,
        "note": (
            "Global tender may be used. Purchase preference for Class I over others applies. "
            "Nodal Ministry may still impose tighter category-specific restrictions."
        ),
    },
}

PURCHASE_PREFERENCE_MARGIN = 20  # percent

SPLIT_ORDER_RULES = {
    "when": (
        "When L1 is a non-local supplier (or Class II) and a local supplier (Class I) "
        "has quoted within the margin of purchase preference."
    ),
    "ratio": "50% to L1, remaining 50% to the eligible local supplier at L1 price.",
    "condition": (
        "Split order is only applicable if the local supplier agrees to match L1 price. "
        "If the local supplier does not agree, the full order goes to L1."
    ),
}

VERIFICATION_METHOD = {
    "self_certification": (
        "The supplier must submit a self-certification of local content "
        "with the bid. False declaration can lead to debarment for 3 years."
    ),
    "nodal_ministry": (
        "Nodal Ministry/Department may prescribe specific verification methods "
        "for their category of goods/services."
    ),
}


def get_mii_answer(amount_lakhs: float | None, query_lower: str) -> str | None:
    """Return a deterministic Make in India answer for common numeric queries."""
    if any(term in query_lower for term in ("local content", "class i", "class ii", "class 1", "class 2")):
        return (
            "**Make in India - Local Content Requirements (PPO 2017)**\n\n"
            "| Supplier Class | Minimum Local Content |\n"
            "|---|---|\n"
            "| **Class I Local Supplier** | >= 50% local content |\n"
            "| **Class II Local Supplier** | >= 20% but < 50% local content |\n"
            "| **Non-Local Supplier** | < 20% local content |\n\n"
            "**Verification:** Suppliers must submit a self-certification of local content with their bid. "
            "False declaration can lead to debarment for 3 years.\n\n"
            "**Purchase Preference Margin:** 20% - if L1 is non-local but a Class I local supplier "
            "has quoted within 20% of L1 price, the Class I supplier gets preference."
        )

    if amount_lakhs is not None and any(term in query_lower for term in ("eligible", "only local", "can bid", "eligib")):
        if amount_lakhs <= 50:
            slab = ELIGIBILITY_BY_VALUE["upto_50_lakh"]
        elif amount_lakhs <= 20000:  # 200 crore
            slab = ELIGIBILITY_BY_VALUE["above_50_lakh_upto_200_cr"]
        else:
            slab = ELIGIBILITY_BY_VALUE["above_200_cr"]

        eligible_str = ", ".join(slab["eligible"])
        return (
            f"**Make in India Eligibility for {slab['range']}**\n\n"
            f"**Eligible Suppliers:** {eligible_str}\n"
            f"**Purchase Preference:** {'Yes' if slab['purchase_preference'] else 'No'}\n\n"
            f"{slab['note']}"
        )

    if any(term in query_lower for term in ("split order", "split quantity", "50%", "l1 is not a local")):
        return (
            "**Split Order Rule (Make in India - PPO 2017)**\n\n"
            f"**When:** {SPLIT_ORDER_RULES['when']}\n\n"
            f"**Ratio:** {SPLIT_ORDER_RULES['ratio']}\n\n"
            f"**Condition:** {SPLIT_ORDER_RULES['condition']}\n\n"
            f"**Purchase Preference Margin:** {PURCHASE_PREFERENCE_MARGIN}%"
        )

    return None
