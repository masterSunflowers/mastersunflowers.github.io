import shutil
from pathlib import Path


def write_post(posts_dir: Path, post_basename: str, content: str) -> Path:
    posts_dir = Path(posts_dir)
    posts_dir.mkdir(parents=True, exist_ok=True)
    path = posts_dir / f"{post_basename}.md"
    path.write_text(content, encoding="utf-8")
    return path


def find_attachment(wiki_path: Path, name: str):
    matches = list(Path(wiki_path).rglob(name))
    return matches[0] if matches else None


def copy_images(wiki_path: Path, images_dir: Path, slug: str, image_names: list) -> None:
    for name in image_names:
        src = find_attachment(wiki_path, name)
        if src is None:
            print(f"[warn] image not found: {name}")
            continue
        dest_dir = Path(images_dir) / slug
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest_dir / name)


def delete_post(posts_dir: Path, images_dir: Path, entry: dict) -> None:
    # Extract filename from entry["post"] and construct full path from posts_dir
    filename = Path(entry["post"]).name
    post_path = posts_dir / filename
    if post_path.exists():
        post_path.unlink()
    img_folder = Path(images_dir) / entry["slug"]
    if img_folder.exists():
        shutil.rmtree(img_folder)
