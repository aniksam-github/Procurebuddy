PROCUREMENT_SLABS = [
    {
        "min": 0,
        "max": 200000,
        "label": "Up to Rs 2,00,000",
        "mode": "Direct Purchase",
        "committee_required": False,
        "committee": None,
    },
    {
        "min": 200001,
        "max": 2500000,
        "label": "Rs 2,00,001 - Rs 25,00,000",
        "mode": "Purchase Committee (PC)",
        "committee_required": True,
        "committee": "Purchase Committee (PC)",
    },
    {
        "min": 2500001,
        "max": 10000000,
        "label": "Rs 25,00,001 - Rs 1,00,00,000",
        "mode": "Limited Tender Enquiry (LTE)",
        "committee_required": True,
        "committee": "Technical & Purchase Committee (T&PC)",
    },
    {
        "min": 10000001,
        "max": None,
        "label": "Above Rs 1,00,00,000",
        "mode": "Open / Global Tender",
        "committee_required": True,
        "committee": "Technical & Purchase Committee (T&PC)",
    },
]
