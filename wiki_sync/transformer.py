import re
import unicodedata
import yaml


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


def build_published_index(published: list, dates: dict) -> dict:
    index: dict = {}
    for note in published:
        ref = f"{dates[note.rel_id]}-{slugify(note.metadata['title'])}"
        index[note.rel_id] = ref
        index[note.basename] = ref
    return index


def _link_replacement(target: str, label: str, index: dict) -> str:
    ref = index.get(target) or index.get(target.split("/")[-1])
    if ref:
        return f"[{label}]({{% post_url {ref} %}})"
    return label


def resolve_wikilinks(body: str, index: dict) -> str:
    def repl(match):
        inner = match.group(1)
        if "|" in inner:
            target, label = inner.split("|", 1)
            target, label = target.strip(), label.strip()
        else:
            target = inner.strip()
            label = target.split("/")[-1]
        return _link_replacement(target, label, index)

    return re.sub(r"(?<!!)\[\[([^\]]+)\]\]", repl, body)


def rewrite_image_embeds(body: str, slug: str):
    images: list = []

    def repl(match):
        name = match.group(1).strip()
        images.append(name)
        return f"![](../images/{slug}/{name})"

    new_body = re.sub(r"!\[\[([^\]]+)\]\]", repl, body)
    return new_body, images


def render_post(frontmatter: dict, body: str) -> str:
    fm = yaml.safe_dump(frontmatter, allow_unicode=True, sort_keys=False).strip()
    return f"---\n{fm}\n---\n\n{body.strip()}\n"
