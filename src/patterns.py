"""Regex patterns and helpers for Mauritanian license plates."""

import re
from typing import Optional, Tuple

# Patterns derived from the previous Streamlit pages
SERIES_PATTERNS = [
    ("Série normale", re.compile(r"^\d{4}\s*[A-Z]{2}\s*\d{2,3}$")),
    ("Série diplomatique (CD)", re.compile(r"^[A-Z]{2,3}\s*CD\s*\d{4}$")),
    ("Série diplomatique (CMD)", re.compile(r"^[A-Z]{2,3}\s*CMD\s*\d{4}$")),
    ("Série diplomatique (CC)", re.compile(r"^[A-Z]{2,3}\s*CC\s*\d{4}$")),
    ("Série ONU", re.compile(r"^ONU(?:\s*CMD)?\s*\d{4}$")),
    ("Série ASNA", re.compile(r"^ASNA(?:\s*CMD)?\s*\d{4}$")),
    ("Série TT", re.compile(r"^[A-Z]\s*\d{5}\s*TT(?:\s*ER)?$")),
    ("Série IT", re.compile(r"^\d{4}\s*IT$")),
    ("Série IF", re.compile(r"^\d{4,5}\s*IF$")),
    ("Série Zone Franche", re.compile(r"^ZFN\s*\d{5}$")),
    ("Série Service Gouvernemental", re.compile(r"^SG\s*\d{5}$")),
    ("Service Conseil Constitutionnel", re.compile(r"^SCC\s*\d{5}$")),
    ("Service parlementaire", re.compile(r"^SP\s*\d{5}$")),
    ("Numérotation Tricycle", re.compile(r"^WT\s*\d{5}$")),
    ("Format spécial (LNNN)", re.compile(r"^[A-Z]\d{3}$")),
]

REGIONS_MAP = {
    "00": "Nouakchott",
    "01": "Hawd Charki",
    "02": "Hawd Karbi",
    "03": "Assaba",
    "04": "Gorgol",
    "05": "Brakna",
    "06": "Trarza",
    "07": "Adrar",
    "08": "Nouadhibou",
    "09": "Teganete",
    "10": "Gidimaka",
    "11": "Tiris Zemmour",
    "12": "Inchiri",
}


def classify_plate(text_norm: str) -> Tuple[str, str]:
    """Return (serie, region) based on normalized plate text."""

    serie: Optional[str] = "Inconnue"
    for name, pattern in SERIES_PATTERNS:
        if pattern.match(text_norm):
            serie = name
            break

    region = "Inconnue"
    if serie == "Série normale":
        match = re.search(r"(\d{2})$", text_norm)
        if match:
            region = REGIONS_MAP.get(match.group(1), "Inconnue")

    return serie or "Inconnue", region

