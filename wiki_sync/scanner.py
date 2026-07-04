from dataclasses import dataclass
from pathlib import Path
import frontmatter


@dataclass
class Note:
    path: Path
    rel_id: str
    basename: str
    metadata: dict
    content: str


def scan_wiki(wiki_path: Path) -> list[Note]:
    notes: list[Note] = []
    for md_path in sorted(wiki_path.rglob("*.md")):
        post = frontmatter.load(md_path)
        if not post.metadata.get("title"):
            print(f"[skip] no title: {md_path}")
            continue
        rel = md_path.relative_to(wiki_path).with_suffix("")
        notes.append(
            Note(
                path=md_path,
                rel_id=rel.as_posix(),
                basename=md_path.stem,
                metadata=dict(post.metadata),
                content=post.content,
            )
        )
    return notes


def filter_published(notes: list[Note]) -> list[Note]:
    return [n for n in notes if n.metadata.get("publish") is True]
