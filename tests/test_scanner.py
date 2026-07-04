from wiki_sync.scanner import scan_wiki, filter_published, Note


def _write(p, text):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_scan_wiki_parses_notes(tmp_path):
    _write(tmp_path / "Notes/A.md", "---\ntitle: A\npublish: true\n---\nBody A\n")
    _write(tmp_path / "Notes/B.md", "---\ntitle: B\n---\nBody B\n")
    notes = scan_wiki(tmp_path)
    by_id = {n.rel_id: n for n in notes}
    assert set(by_id) == {"Notes/A", "Notes/B"}
    assert by_id["Notes/A"].basename == "A"
    assert by_id["Notes/A"].content.strip() == "Body A"


def test_scan_wiki_skips_notes_without_title(tmp_path):
    _write(tmp_path / "Notes/NoTitle.md", "---\ntype: concept-note\n---\nBody\n")
    assert scan_wiki(tmp_path) == []


def test_filter_published(tmp_path):
    _write(tmp_path / "A.md", "---\ntitle: A\npublish: true\n---\nx\n")
    _write(tmp_path / "B.md", "---\ntitle: B\n---\ny\n")
    published = filter_published(scan_wiki(tmp_path))
    assert [n.basename for n in published] == ["A"]
