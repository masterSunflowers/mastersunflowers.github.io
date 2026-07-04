import argparse
import datetime
import subprocess
import sys
from pathlib import Path

from wiki_sync.config import load_config
from wiki_sync.scanner import scan_wiki, filter_published
from wiki_sync.manifest import load_manifest, save_manifest, stamp_date, removed_entries
from wiki_sync.transformer import build_published_index, transform_note
from wiki_sync.writer import write_post, copy_images, delete_post

MANIFEST_PATH = Path(".wiki-sync.json")


def run(dry_run: bool, push: bool) -> None:
    config = load_config()
    if not config.vault_wiki_path.exists():
        raise SystemExit(f"Vault path not found: {config.vault_wiki_path}")

    today = datetime.date.today().isoformat()
    manifest = load_manifest(MANIFEST_PATH)
    published = filter_published(scan_wiki(config.vault_wiki_path))

    dates = {n.rel_id: stamp_date(n, manifest, today) for n in published}
    index = build_published_index(published, dates)

    new_manifest: dict = {}
    for note in published:
        date = dates[note.rel_id]
        basename, content, images = transform_note(note, date, index, config)
        post_rel = f"{config.posts_dir.as_posix()}/{basename}.md"
        print(f"[sync] {note.rel_id} -> {post_rel}")
        if not dry_run:
            write_post(config.posts_dir, basename, content)
            copy_images(config.vault_wiki_path, config.images_dir,
                        basename.split("-", 3)[-1], images)
        new_manifest[note.rel_id] = {"slug": basename.split("-", 3)[-1],
                                     "date": date, "post": post_rel}

    for orphan in removed_entries(manifest, {n.rel_id for n in published}):
        print(f"[remove] {orphan['post']}")
        if not dry_run:
            delete_post(config.posts_dir, config.images_dir, orphan)

    if dry_run:
        print("[dry-run] no files written, no commit.")
        return

    save_manifest(MANIFEST_PATH, new_manifest)
    _git_sync(config.git_branch, push)


def _git_sync(branch: str, push: bool) -> None:
    subprocess.run(["git", "add", "_posts", "images", str(MANIFEST_PATH)], check=True)
    staged = subprocess.run(["git", "diff", "--cached", "--quiet"]).returncode
    if staged == 0:
        print("[git] nothing to sync.")
        return
    msg = f"chore: sync wiki {datetime.date.today().isoformat()}"
    subprocess.run(["git", "commit", "-m", msg], check=True)
    if push:
        subprocess.run(["git", "push", "origin", branch], check=True)
        print("[git] pushed.")
    else:
        print("[git] committed locally (--no-push).")


def main() -> None:
    # Note titles/paths contain Vietnamese; force UTF-8 stdout so progress
    # prints don't crash on a Windows console using a legacy code page (cp1252).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Sync LLM Wiki notes to the Jekyll site.")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing or committing.")
    parser.add_argument("--no-push", action="store_true", help="Convert and commit, but do not push.")
    args = parser.parse_args()
    run(dry_run=args.dry_run, push=not args.no_push)


if __name__ == "__main__":
    main()
