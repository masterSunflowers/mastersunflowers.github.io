import re
import unicodedata


def slugify(title: str) -> str:
    text = title.replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def extract_category(metadata: dict, default: str) -> str:
    topics = metadata.get("topics") or []
    if topics:
        first = str(topics[0])
        m = re.search(r"\[\[([^\]]+)\]\]", first)
        inner = m.group(1) if m else first
        label = inner.split("|")[-1].strip()
        if label:
            return label
    return default
