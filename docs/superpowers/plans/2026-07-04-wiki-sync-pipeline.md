# LLM Wiki → Site Sync Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `python sync.py` command that converts LLM Wiki notes marked `publish: true` into Jekyll posts, then commits and pushes so GitHub Pages rebuilds.

**Architecture:** A small Python package `wiki_sync/` with focused modules (config, scanner, transformer, manifest, writer) orchestrated by `sync.py`. Pure text-transform functions are unit-tested; a JSON manifest keeps post URLs stable and drives removals.

**Tech Stack:** Python 3.11, `python-frontmatter`, `PyYAML`, `pytest`, git CLI.

## Global Constraints

- Run all commands from repo root: `D:/Workspace/mastersunflowers.github.io`.
- Use `python` (not `py`) — this is a Windows + Git Bash environment.
- Git branch: `main`.
- Vault wiki path (verified): `G:/My Drive/DriveSyncFiles/Obsidian/Knowledge/LLM Wiki`.
- Dependencies limited to: `python-frontmatter`, `PyYAML`, `pytest`. No other third-party libs.
- Jekyll internal links use the Liquid tag `{% post_url YYYY-MM-DD-slug %}` so links stay correct regardless of the site's permalink scheme.
- Date format everywhere: `YYYY-MM-DD` strings.
- Every module lives under `wiki_sync/`; tests under `tests/`.

---

## File Structure

```
mastersunflowers.github.io/
├── sync.py                     # entry point: CLI, orchestration, git
├── sync_config.yml             # paths + defaults
├── requirements.txt            # python-frontmatter, PyYAML, pytest
├── wiki_sync/
│   ├── __init__.py
│   ├── config.py               # Config dataclass + load_config
│   ├── scanner.py              # Note dataclass, scan_wiki, filter_published
│   ├── manifest.py             # load/save manifest, stamp_date, removed_entries
│   ├── transformer.py          # slugify, category, wikilinks, images, frontmatter, render
│   └── writer.py               # write_post, copy_images, delete_post
├── tests/
│   ├── test_config.py
│   ├── test_scanner.py
│   ├── test_manifest.py
│   ├── test_transformer.py
│   └── test_writer.py
└── .wiki-sync.json             # manifest (committed)
```

---

### Task 1: Project scaffold + config loader

**Files:**
- Create: `requirements.txt`, `sync_config.yml`, `wiki_sync/__init__.py`, `wiki_sync/config.py`, `tests/test_config.py`

**Interfaces:**
- Produces: `Config` dataclass with fields `vault_wiki_path: Path`, `posts_dir: Path`, `images_dir: Path`, `default_author: str`, `default_category: str`, `git_branch: str`; and `load_config(path: str = "sync_config.yml") -> Config`.

- [ ] **Step 1: Create dependency + config files**

`requirements.txt`:
```
python-frontmatter==1.1.0
PyYAML==6.0.2
pytest==8.3.3
```

`sync_config.yml`:
```yaml
vault_wiki_path: "G:/My Drive/DriveSyncFiles/Obsidian/Knowledge/LLM Wiki"
posts_dir: "_posts"
images_dir: "images"
default_author: "Thieu Luu"
default_category: "LLM Wiki"
git_branch: "main"
```

`wiki_sync/__init__.py`:
```python
```

- [ ] **Step 2: Install dependencies**

Run: `python -m pip install -r requirements.txt`
Expected: installs python-frontmatter, PyYAML, pytest without error.

- [ ] **Step 3: Write the failing test**

`tests/test_config.py`:
```python
from pathlib import Path
from wiki_sync.config import load_config


def test_load_config_reads_fields(tmp_path):
    cfg_file = tmp_path / "sync_config.yml"
    cfg_file.write_text(
        'vault_wiki_path: "X/Wiki"\n'
        'posts_dir: "_posts"\n'
        'images_dir: "images"\n'
        'default_author: "Thieu Luu"\n'
        'default_category: "LLM Wiki"\n'
        'git_branch: "main"\n',
        encoding="utf-8",
    )
    cfg = load_config(str(cfg_file))
    assert cfg.vault_wiki_path == Path("X/Wiki")
    assert cfg.posts_dir == Path("_posts")
    assert cfg.default_author == "Thieu Luu"
    assert cfg.git_branch == "main"
```

- [ ] **Step 4: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'wiki_sync.config'`.

- [ ] **Step 5: Write minimal implementation**

`wiki_sync/config.py`:
```python
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class Config:
    vault_wiki_path: Path
    posts_dir: Path
    images_dir: Path
    default_author: str
    default_category: str
    git_branch: str


def load_config(path: str = "sync_config.yml") -> Config:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(
        vault_wiki_path=Path(data["vault_wiki_path"]),
        posts_dir=Path(data["posts_dir"]),
        images_dir=Path(data["images_dir"]),
        default_author=data["default_author"],
        default_category=data["default_category"],
        git_branch=data["git_branch"],
    )
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/test_config.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add requirements.txt sync_config.yml wiki_sync/__init__.py wiki_sync/config.py tests/test_config.py
git commit -m "feat: scaffold wiki_sync package and config loader"
```

---

### Task 2: Slug + category helpers

**Files:**
- Create: `wiki_sync/transformer.py`, `tests/test_transformer.py`

**Interfaces:**
- Produces: `slugify(title: str) -> str` (Vietnamese-diacritic-safe kebab-case); `extract_category(metadata: dict, default: str) -> str` (reads label from `topics[0]` wikilink, else `default`).

- [ ] **Step 1: Write the failing tests**

`tests/test_transformer.py`:
```python
from wiki_sync.transformer import slugify, extract_category


def test_slugify_strips_vietnamese_diacritics():
    assert slugify("Hồi quy tuyến tính") == "hoi-quy-tuyen-tinh"


def test_slugify_handles_symbols_and_spacing():
    assert slugify("Continuous Control - PPO, SAC, TD3") == "continuous-control-ppo-sac-td3"


def test_extract_category_reads_topic_label():
    md = {"topics": ["[[Knowledge/LLM Wiki/Topics/RL for Robotics|RL for Robotics]]"]}
    assert extract_category(md, "LLM Wiki") == "RL for Robotics"


def test_extract_category_falls_back_to_default():
    assert extract_category({}, "LLM Wiki") == "LLM Wiki"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_transformer.py -v`
Expected: FAIL with `ImportError: cannot import name 'slugify'`.

- [ ] **Step 3: Write minimal implementation**

`wiki_sync/transformer.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_transformer.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add wiki_sync/transformer.py tests/test_transformer.py
git commit -m "feat: add slugify and category extraction"
```

---

### Task 3: Vault scanner

**Files:**
- Create: `wiki_sync/scanner.py`, `tests/test_scanner.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces:
  - `Note` dataclass: `path: Path`, `rel_id: str` (vault-relative path without `.md`, forward slashes), `basename: str`, `metadata: dict`, `content: str` (body only).
  - `scan_wiki(wiki_path: Path) -> list[Note]` — every `.md` with a `title`; skips (and prints) notes lacking `title`.
  - `filter_published(notes: list[Note]) -> list[Note]` — keeps `metadata.get("publish") is True`.

- [ ] **Step 1: Write the failing tests**

`tests/test_scanner.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_scanner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'wiki_sync.scanner'`.

- [ ] **Step 3: Write minimal implementation**

`wiki_sync/scanner.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_scanner.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add wiki_sync/scanner.py tests/test_scanner.py
git commit -m "feat: add vault scanner with publish filter"
```

---

### Task 4: Manifest (stable dates + removals)

**Files:**
- Create: `wiki_sync/manifest.py`, `tests/test_manifest.py`

**Interfaces:**
- Consumes: `Note` from `wiki_sync.scanner`.
- Produces:
  - `load_manifest(path: Path) -> dict` — `{rel_id: {"slug": str, "date": str, "post": str}}`; `{}` if file missing.
  - `save_manifest(path: Path, manifest: dict) -> None`.
  - `stamp_date(note: Note, manifest: dict, today: str) -> str` — reuse existing manifest date if present, else priority `created` > `wiki_updated` > `today`, returned as `YYYY-MM-DD`.
  - `removed_entries(manifest: dict, current_rel_ids: set[str]) -> list[dict]` — manifest entries whose `rel_id` is not in `current_rel_ids`.

- [ ] **Step 1: Write the failing tests**

`tests/test_manifest.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_manifest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'wiki_sync.manifest'`.

- [ ] **Step 3: Write minimal implementation**

`wiki_sync/manifest.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_manifest.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add wiki_sync/manifest.py tests/test_manifest.py
git commit -m "feat: add manifest with stable dates and orphan detection"
```

---

### Task 5: Body transforms — wikilinks, images, render

**Files:**
- Modify: `wiki_sync/transformer.py`
- Modify: `tests/test_transformer.py`

**Interfaces:**
- Consumes: `slugify` (Task 2).
- Produces:
  - `build_published_index(published: list, dates: dict) -> dict[str, str]` — maps each published note's `rel_id` AND `basename` to its post ref `"YYYY-MM-DD-slug"`. `dates` is `{rel_id: date_str}`.
  - `resolve_wikilinks(body: str, index: dict) -> str` — `[[target|label]]` → `[label]({% post_url ref %})` when `target` (its `rel_id` or basename) is in `index`, else `label`; `[[target]]` uses the basename of `target` as the label.
  - `rewrite_image_embeds(body: str, slug: str) -> tuple[str, list[str]]` — `![[name.ext]]` → `![](../images/<slug>/name.ext)`, returns list of embedded image filenames.
  - `render_post(frontmatter: dict, body: str) -> str` — YAML frontmatter block + body.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_transformer.py`:
```python
from wiki_sync.transformer import (
    build_published_index,
    resolve_wikilinks,
    rewrite_image_embeds,
    render_post,
)
from wiki_sync.scanner import Note
from pathlib import Path


def _pub(rel_id, basename, title):
    return Note(path=Path("x"), rel_id=rel_id, basename=basename,
                metadata={"title": title, "publish": True}, content="")


def test_build_published_index_maps_relid_and_basename():
    notes = [_pub("Knowledge/LLM Wiki/Notes/DQN", "DQN", "Deep Q-Network")]
    idx = build_published_index(notes, {"Knowledge/LLM Wiki/Notes/DQN": "2026-06-27"})
    assert idx["Knowledge/LLM Wiki/Notes/DQN"] == "2026-06-27-deep-q-network"
    assert idx["DQN"] == "2026-06-27-deep-q-network"


def test_resolve_wikilink_to_published_becomes_posturl():
    idx = {"DQN": "2026-06-27-deep-q-network"}
    out = resolve_wikilinks("see [[DQN|Deep Q-Network]] here", idx)
    assert out == "see [Deep Q-Network]({% post_url 2026-06-27-deep-q-network %}) here"


def test_resolve_wikilink_unpublished_becomes_plain_label():
    out = resolve_wikilinks("see [[Some/Path|My Label]] now", {})
    assert out == "see My Label now"


def test_resolve_wikilink_without_alias_uses_basename():
    out = resolve_wikilinks("ref [[Knowledge/Notes/Foo]] end", {})
    assert out == "ref Foo end"


def test_rewrite_image_embeds():
    body, imgs = rewrite_image_embeds("text ![[diagram.png]] more", "my-slug")
    assert body == "text ![](../images/my-slug/diagram.png) more"
    assert imgs == ["diagram.png"]


def test_render_post_emits_frontmatter_and_body():
    out = render_post({"title": "A", "layout": "post"}, "Body here")
    assert out.startswith("---\n")
    assert "title: A" in out
    assert out.rstrip().endswith("Body here")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_transformer.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_published_index'`.

- [ ] **Step 3: Write minimal implementation**

Append to `wiki_sync/transformer.py`:
```python
import yaml


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

    return re.sub(r"\[\[([^\]]+)\]\]", repl, body)


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_transformer.py -v`
Expected: PASS (10 passed).

- [ ] **Step 5: Commit**

```bash
git add wiki_sync/transformer.py tests/test_transformer.py
git commit -m "feat: add wikilink resolution, image rewrite, post rendering"
```

---

### Task 6: Frontmatter mapping + full note transform

**Files:**
- Modify: `wiki_sync/transformer.py`
- Modify: `tests/test_transformer.py`

**Interfaces:**
- Consumes: `slugify`, `extract_category`, `resolve_wikilinks`, `rewrite_image_embeds`, `render_post` (Tasks 2 & 5); `Config` (Task 1); `Note` (Task 3).
- Produces:
  - `map_frontmatter(note: Note, date: str, config) -> dict` — `{title, author, date, category, layout}`; author from `note.metadata.get("author")` else `config.default_author`.
  - `transform_note(note: Note, date: str, index: dict, config) -> tuple[str, str, list[str]]` — returns `(post_basename, content, image_names)` where `post_basename = "<date>-<slug>"`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_transformer.py`:
```python
from wiki_sync.transformer import map_frontmatter, transform_note
from wiki_sync.config import Config


def _cfg():
    return Config(
        vault_wiki_path=Path("wiki"), posts_dir=Path("_posts"), images_dir=Path("images"),
        default_author="Thieu Luu", default_category="LLM Wiki", git_branch="main",
    )


def _note_full(meta, content):
    return Note(path=Path("x.md"), rel_id="Notes/T", basename="T", metadata=meta, content=content)


def test_map_frontmatter_defaults():
    note = _note_full({"title": "Deep Q-Network",
                       "topics": ["[[Topics/RL for Robotics|RL for Robotics]]"]}, "")
    fm = map_frontmatter(note, "2026-06-27", _cfg())
    assert fm == {"title": "Deep Q-Network", "author": "Thieu Luu",
                  "date": "2026-06-27", "category": "RL for Robotics", "layout": "post"}


def test_transform_note_produces_basename_and_content():
    note = _note_full({"title": "Deep Q-Network"}, "Body with [[T|self]] link")
    idx = {"Notes/T": "2026-06-27-deep-q-network", "T": "2026-06-27-deep-q-network"}
    basename, content, imgs = transform_note(note, "2026-06-27", idx, _cfg())
    assert basename == "2026-06-27-deep-q-network"
    assert "layout: post" in content
    assert "{% post_url 2026-06-27-deep-q-network %}" in content
    assert imgs == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_transformer.py -v`
Expected: FAIL with `ImportError: cannot import name 'map_frontmatter'`.

- [ ] **Step 3: Write minimal implementation**

Append to `wiki_sync/transformer.py`:
```python
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
    content = render_post(map_frontmatter(note, date, config), body)
    return post_basename, content, images
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_transformer.py -v`
Expected: PASS (12 passed).

- [ ] **Step 5: Commit**

```bash
git add wiki_sync/transformer.py tests/test_transformer.py
git commit -m "feat: add frontmatter mapping and full note transform"
```

---

### Task 7: Writer

**Files:**
- Create: `wiki_sync/writer.py`, `tests/test_writer.py`

**Interfaces:**
- Consumes: nothing structural from other tasks (operates on strings/paths).
- Produces:
  - `write_post(posts_dir: Path, post_basename: str, content: str) -> Path` — writes `posts_dir/<post_basename>.md`, returns the path.
  - `find_attachment(wiki_path: Path, name: str) -> Path | None` — first `.rglob` match for `name`, else `None`.
  - `copy_images(wiki_path: Path, images_dir: Path, slug: str, image_names: list) -> None` — copies each found attachment to `images_dir/<slug>/`; prints a warning for missing ones.
  - `delete_post(posts_dir: Path, images_dir: Path, entry: dict) -> None` — removes `entry["post"]` and the `images_dir/<slug>` folder if present.

- [ ] **Step 1: Write the failing tests**

`tests/test_writer.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_writer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'wiki_sync.writer'`.

- [ ] **Step 3: Write minimal implementation**

`wiki_sync/writer.py`:
```python
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
    post_path = Path(entry["post"])
    if post_path.exists():
        post_path.unlink()
    img_folder = Path(images_dir) / entry["slug"]
    if img_folder.exists():
        shutil.rmtree(img_folder)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_writer.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add wiki_sync/writer.py tests/test_writer.py
git commit -m "feat: add writer for posts, images, and deletions"
```

---

### Task 8: Orchestrator `sync.py` (CLI + git) and integration run

**Files:**
- Create: `sync.py`

**Interfaces:**
- Consumes: `load_config` (Task 1); `scan_wiki`, `filter_published` (Task 3); `load_manifest`, `save_manifest`, `stamp_date`, `removed_entries` (Task 4); `build_published_index`, `transform_note` (Tasks 5–6); `write_post`, `copy_images`, `delete_post` (Task 7).
- Produces: an executable `python sync.py [--dry-run] [--no-push]` entry point.

- [ ] **Step 1: Write the orchestrator**

`sync.py`:
```python
import argparse
import datetime
import subprocess
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
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    if not status.stdout.strip():
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
    parser = argparse.ArgumentParser(description="Sync LLM Wiki notes to the Jekyll site.")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing or committing.")
    parser.add_argument("--no-push", action="store_true", help="Convert and commit, but do not push.")
    args = parser.parse_args()
    run(dry_run=args.dry_run, push=not args.no_push)


if __name__ == "__main__":
    main()
```

Note: `basename.split("-", 3)[-1]` recovers the slug from `YYYY-MM-DD-slug` (dates have exactly 3 hyphens before the slug).

- [ ] **Step 2: Run the full test suite**

Run: `python -m pytest -v`
Expected: PASS (all tests from Tasks 1–7, 24 passed).

- [ ] **Step 3: Mark two vault notes for publishing (manual)**

In the Obsidian vault, add `publish: true` to the frontmatter of two finished notes, e.g. `Knowledge/LLM Wiki/Notes/Reinforcement learning.md` and `Knowledge/LLM Wiki/Notes/Hồi quy tuyến tính.md`.

- [ ] **Step 4: Integration dry-run**

Run: `python sync.py --dry-run`
Expected: prints `[sync] Knowledge/LLM Wiki/Notes/Reinforcement learning -> _posts/<date>-reinforcement-learning.md` and the linear-regression line, then `[dry-run] no files written, no commit.` No files created (verify `git status` is clean).

- [ ] **Step 5: Real run without push, inspect output**

Run: `python sync.py --no-push`
Then inspect: `python -m pytest -q` still passes, and open the generated file under `_posts/` to confirm frontmatter (`layout: post`, correct `category`) and that `$$...$$` math and any resolved `{% post_url %}` links look right.

- [ ] **Step 6: Commit the pipeline**

```bash
git add sync.py .wiki-sync.json _posts images
git commit -m "feat: add sync.py orchestrator with dry-run and push flags"
```

- [ ] **Step 7: Push (final)**

Run: `python sync.py` (or `git push origin main`)
Expected: GitHub Pages rebuilds; the two notes appear on the live site within a minute.

---

## Notes for the implementer

- If `python -m pip install` reports that `python-frontmatter` imports as `frontmatter`, that is expected — the package name and import name differ.
- Windows line endings: git may warn `LF will be replaced by CRLF`. Harmless.
- The `_posts` and `images` directories already exist in the repo; the writer's `mkdir(parents=True, exist_ok=True)` is safe.
- Do not commit `_site/` or `.jekyll-cache/` — already covered by `.gitignore`.
