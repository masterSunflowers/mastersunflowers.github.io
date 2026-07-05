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


# kramdown uses GFM input, so a bare `|` — even inside `$...$` math — is read
# as a table-column delimiter and shreds the formula into a stray HTML table.
# Obsidian renders the same source fine, so notes are authored with plain `|`.
# Rewrite pipes *inside math spans only* to TeX equivalents (`||` -> \|, a
# single `|` -> \mid) while leaving real Markdown table pipes untouched.
_MATH_SPAN = re.compile(r"\$\$.+?\$\$|\$.+?\$", re.DOTALL)


def _escape_span(match: "re.Match") -> str:
    span = match.group(0)
    span = re.sub(r"(?<!\\)\|\|", r"\\|", span)      # norm / KL: || -> \|
    span = re.sub(r"(?<!\\)\|", r"\\mid ", span)     # conditional / abs: | -> \mid
    return span


def escape_math_pipes(body: str) -> str:
    return _MATH_SPAN.sub(_escape_span, body)


def render_post(frontmatter: dict, body: str) -> str:
    fm = yaml.safe_dump(frontmatter, allow_unicode=True, sort_keys=False).strip()
    return f"---\n{fm}\n---\n\n{body.strip()}\n"


def map_frontmatter(note, date: str, config) -> dict:
    return {
        "title": note.metadata["title"],
        "author": note.metadata.get("author", config.default_author),
        "date": date,
        "category": extract_category(note.metadata, config.default_category),
        "layout": "post",
    }


def transform_note(note, date: str, index: dict, config):
    slug = slugify(note.metadata["title"])
    post_basename = f"{date}-{slug}"
    body = resolve_wikilinks(note.content, index)
    body, images = rewrite_image_embeds(body, slug)
    body = escape_math_pipes(body)
    content = render_post(map_frontmatter(note, date, config), body)
    return post_basename, content, images
