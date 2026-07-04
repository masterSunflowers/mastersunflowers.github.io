from pathlib import Path
from wiki_sync.manifest import load_manifest, save_manifest, stamp_date, removed_entries
from wiki_sync.scanner import Note


def _note(meta):
    return Note(path=Path("x.md"), rel_id="Notes/X", basename="X", metadata=meta, content="")


def test_load_missing_manifest_returns_empty(tmp_path):
    assert load_manifest(tmp_path / "nope.json") == {}


def test_save_and_load_roundtrip(tmp_path):
    p = tmp_path / "m.json"
    data = {"Notes/X": {"slug": "x", "date": "2026-01-02", "post": "_posts/2026-01-02-x.md"}}
    save_manifest(p, data)
    assert load_manifest(p) == data


def test_stamp_date_prefers_existing_manifest():
    manifest = {"Notes/X": {"slug": "x", "date": "2025-05-05", "post": "p"}}
    note = _note({"title": "X", "created": "2026-06-01"})
    assert stamp_date(note, manifest, "2026-07-04") == "2025-05-05"


def test_stamp_date_priority_created_then_updated_then_today():
    assert stamp_date(_note({"created": "2026-06-01", "wiki_updated": "2026-06-03"}), {}, "2026-07-04") == "2026-06-01"
    assert stamp_date(_note({"wiki_updated": "2026-06-03"}), {}, "2026-07-04") == "2026-06-03"
    assert stamp_date(_note({}), {}, "2026-07-04") == "2026-07-04"


def test_removed_entries_finds_orphans():
    manifest = {
        "Notes/X": {"slug": "x", "date": "2026-01-01", "post": "_posts/2026-01-01-x.md"},
        "Notes/Y": {"slug": "y", "date": "2026-01-02", "post": "_posts/2026-01-02-y.md"},
    }
    orphans = removed_entries(manifest, {"Notes/X"})
    assert orphans == [{"slug": "y", "date": "2026-01-02", "post": "_posts/2026-01-02-y.md"}]
