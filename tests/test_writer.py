from pathlib import Path
from wiki_sync.writer import write_post, find_attachment, copy_images, delete_post


def test_write_post_creates_file(tmp_path):
    posts = tmp_path / "_posts"
    path = write_post(posts, "2026-06-27-x", "---\ntitle: X\n---\nBody\n")
    assert path == posts / "2026-06-27-x.md"
    assert path.read_text(encoding="utf-8").startswith("---")


def test_find_attachment(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub/pic.png").write_bytes(b"img")
    assert find_attachment(tmp_path, "pic.png") == tmp_path / "sub/pic.png"
    assert find_attachment(tmp_path, "missing.png") is None


def test_copy_images_copies_found(tmp_path):
    wiki = tmp_path / "wiki"; wiki.mkdir()
    (wiki / "pic.png").write_bytes(b"img")
    images = tmp_path / "images"
    copy_images(wiki, images, "my-slug", ["pic.png"])
    assert (images / "my-slug/pic.png").read_bytes() == b"img"


def test_delete_post_removes_file_and_images(tmp_path):
    posts = tmp_path / "_posts"; posts.mkdir()
    (posts / "2026-01-01-y.md").write_text("x", encoding="utf-8")
    images = tmp_path / "images"; (images / "y").mkdir(parents=True)
    (images / "y/pic.png").write_bytes(b"i")
    delete_post(posts, images,
                {"slug": "y", "date": "2026-01-01", "post": "_posts/2026-01-01-y.md"})
    assert not (posts / "2026-01-01-y.md").exists()
    assert not (images / "y").exists()
