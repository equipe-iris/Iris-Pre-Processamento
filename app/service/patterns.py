import pandas as pd
import re
from bs4 import BeautifulSoup


patterns = [
    {"name": "remove_html_tags", "func": lambda text: BeautifulSoup(text, "html.parser").get_text()},
    {"name": "remove_color_tags", "pattern": r'\{.*?\}', "flags": 0},
    {"name": "remove_urls", "pattern": r'https?://[^\s!]+', "flags": 0},
    {"name": "remove_signature", "pattern": r'atenciosamente.*$', "flags": re.IGNORECASE},
    {"name": "remove_id_codes", "pattern": r'\b(?=[0-9A-Za-z:-]*:[0-9A-Za-z:-]*)(?=[0-9A-Za-z:-]*-[0-9A-Za-z:-]*)[0-9A-Za-z:-]+\b', "flags": 0},
    {"name": "remove_datetime", "pattern": r'\b\d{1,2}/[A-Za-z]{3}/\d{2}\s+\d{1,2}:\d{2}\s+(?:AM|PM)\b', "flags": re.IGNORECASE},
    {"name": "remove_image_refs", "pattern": r"!.*?\|thumbnail!", "flags": 0},
    {"name": "remove_newlines", "pattern": r'[\r\n]+', "replacement": " ", "flags": 0},
    {"name": "remove_pdf_filenames", "pattern": r'\b\S+\.pdf\b', "flags": re.IGNORECASE},
    {"name": "remove_custom_words", "pattern": r'\b(?:content|attrs|At.te)\b', "flags": re.IGNORECASE},
    {"name": "remove_special_characters", "pattern": r'[^A-Za-zÀ-ÖØ-öø-ÿ0-9\s,:!?.@;\-]', "replacement": " ", "flags": 0},
    {"name": "remove_extra_spaces", "pattern": r'\s{2,}', "replacement": " ", "flags": 0},
    {"name": "remove_isolated_special_chars", "pattern": r'(?<![A-Za-z0-9])([^A-Za-z0-9\s]+)(?![A-Za-z0-9])', "flags": 0},
    {"name": "remove_numeric_codes", "pattern": r'\d+', "replacement": " ", "flags": 0},
    {"name": "remove_media_refs", "pattern": r'!(?=\S)(?:.*?)(?<=\S)!', "flags": 0},
    {"name": "remove_gccode_blocks", "pattern": r'gccode.*?!', "flags": 0},
    {"name": "remove_whatsapp_video", "pattern": r'Vídeo do WhatsApp de \d{4}-\d{2}-\d{2} s \d{1,2}\.\d{2}\.\d{2} [0-9A-Za-z]+\.mp4', "flags": 0},
    {"name": "remove_square_brackets", "pattern": r'\[.*?\]', "flags": 0},
    {"name": "remove_file_sizes", "pattern": r'\b\d+(?:\.\d+)?\s*(?:kB|MB)\b', "flags": 0},
    {"name": "remove_bin_filenames", "pattern": r'\b\S+?\.bin\b', "flags": 0},
    {"name": "remove_uuids", "pattern": r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b', "flags": 0},
    {"name": "replace_semicolons", "pattern": r';', "replacement": " ", "flags": 0},
]

def normalizeText(text, patterns):
    if pd.isnull(text):
        return text

    for pattern in patterns:
        if "func" in pattern:
            text = pattern["func"](text)
        else:
            replacement = pattern.get("replacement", "")
            text = re.sub(pattern["pattern"], replacement, text, flags=pattern.get("flags", 0))

    return text.strip()
