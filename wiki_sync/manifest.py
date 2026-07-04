import json
from pathlib import Path
from wiki_sync.scanner import Note


def load_manifest(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(path: Path, manifest: dict) -> None:
    Path(path).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def stamp_date(note: Note, manifest: dict, today: str) -> str:
    existing = manifest.get(note.rel_id)
    if existing and existing.get("date"):
        return existing["date"]
    for key in ("created", "wiki_updated"):
        val = note.metadata.get(key)
        if val:
            return str(val)[:10]
    return today


def removed_entries(manifest: dict, current_rel_ids: set) -> list:
    return [entry for rel_id, entry in manifest.items() if rel_id not in current_rel_ids]
