# LLM Wiki → Site Sync Pipeline — Design

**Date:** 2026-07-04
**Status:** Approved (design)

## 1. Mục tiêu

Tự động chuyển các note kiến thức được đánh dấu công khai trong **LLM Wiki** (Obsidian vault) thành bài viết Jekyll trên trang cá nhân public **mastersunflowers.github.io**, rồi commit & push để GitHub Pages tự build.

- **Nguồn:** `G:\My Drive\DriveSyncFiles\Obsidian\Knowledge\LLM Wiki\**\*.md`
- **Đích:** repo site `mastersunflowers.github.io`, thư mục `_posts/`
- **Pipeline sống trong:** repo site (script + config version-control cùng nơi xuất bản)

## 2. Quyết định thiết kế (đã chốt)

| Vấn đề | Quyết định |
|---|---|
| Chọn note để public | Cờ `publish: true` trong frontmatter note |
| Mức tự động | Một script convert + `git commit` + `git push` |
| Vị trí code | Trong repo site |
| Wikilink `[[...]]` | Trỏ tới note cũng public → hyperlink; còn lại → text thường (không gãy) |
| Author | Mặc định `Thieu Luu` |
| Category | Suy từ tên `topics` của note |
| Ảnh embed | Tối giản: copy khi gặp, sửa path |
| Ngôn ngữ công cụ | Python |

## 3. Kiến trúc

Một lệnh `python sync.py` chạy 4 bước tách bạch, mỗi bước một module có thể test độc lập:

```
vault LLM Wiki ──▶ [1 Scan+Filter] ──▶ [2 Transform] ──▶ [3 Write] ──▶ [4 Git] ──▶ GitHub Pages
                        │                    ▲
                        └── published-set ───┘
                                 manifest (.wiki-sync.json)
```

### 3.1 Scan & Filter (`scanner.py`)
- Duyệt đệ quy `LLM Wiki/**/*.md`, parse YAML frontmatter (`python-frontmatter`).
- Giữ note có `publish: true`.
- Trả về danh sách note + dựng **published-set**: map `{tên note / đường dẫn vault → slug/URL post}` để bước Transform phân giải wikilink.
- Note thiếu `title` → skip + log cảnh báo.

### 3.2 Transform (`transformer.py`) — thuần, dễ test
Với mỗi note:

**Frontmatter Obsidian → Jekyll:**
```yaml
# Ra:
title:    <title>
author:   Thieu Luu            # mặc định, override nếu note có author
date:     <date đã stamp>       # xem §3.5 ổn định URL
category: <tên topic đầu tiên>  # từ topics[]; fallback "LLM Wiki"
layout:   post
```
Bỏ các field Obsidian-only: `type, status, wiki, topics, tags, summary, wiki_updated, parent, aliases, publish`.

**Body:**
- **Wikilink** `[[Path|nhãn]]` hoặc `[[Path]]`:
  - Đích ∈ published-set → `[nhãn]({{ site.baseurl }}/<url post>)`.
  - Ngược lại → chỉ giữ `nhãn` (phần sau `|`, hoặc basename nếu không có `|`).
- **Ảnh embed** `![[file.png]]` → copy `file.png` sang `images/<slug>/file.png`, thay bằng `![](../images/<slug>/file.png)`. Ảnh không tìm thấy → giữ text + log.
- **LaTeX** `$...$`, `$$...$$` → giữ nguyên (kramdown + MathJax hai bên tương thích).
- Callout/emoji Obsidian → giữ nguyên markdown (theme render được phần cơ bản).

**Slug:** kebab-case từ title (bỏ dấu tiếng Việt cho URL sạch).

### 3.3 Write (`writer.py`)
- Ghi `_posts/<date>-<slug>.md`.
- Ghi ảnh đã copy vào `images/<slug>/`.
- Idempotent: ghi đè file cùng tên.

### 3.4 Git (`sync.py` entry)
- `git add _posts/ images/ .wiki-sync.json`
- `git commit -m "chore: sync wiki <YYYY-MM-DD>"` (skip nếu không có thay đổi).
- `git push origin <branch>`.
- Lỗi git → dừng, để working tree ở trạng thái sạch (không commit dở).

### 3.5 Manifest & đồng bộ (`.wiki-sync.json`)
File JSON ở gốc repo site, map:
```json
{ "<đường dẫn vault note>": { "slug": "...", "date": "YYYY-MM-DD", "post": "_posts/..." } }
```
Vai trò:
- **URL ổn định:** `date` được stamp **một lần** ở lần sync đầu (ưu tiên `created` > `wiki_updated` > hôm nay), các lần sau tái dùng từ manifest → sửa note không đổi URL, không tạo post mồ côi.
- **Gỡ bài:** note bỏ `publish:true` hoặc bị xoá khỏi vault → không còn trong lần scan này nhưng còn trong manifest → xoá post + thư mục ảnh tương ứng, gỡ khỏi manifest.

## 4. Giao diện dòng lệnh

```
python sync.py            # convert + commit + push
python sync.py --dry-run  # in kế hoạch (thêm/sửa/xoá bài), KHÔNG ghi/commit/push
python sync.py --no-push  # convert + commit, KHÔNG push (tự review trước)
```

## 5. Cấu hình (`sync_config.yml` hoặc đầu file)

```yaml
vault_wiki_path: "G:/My Drive/DriveSyncFiles/Obsidian/Knowledge/LLM Wiki"
posts_dir: "_posts"
images_dir: "images"
default_author: "Thieu Luu"
default_category: "LLM Wiki"
git_branch: "main"
```

## 6. Xử lý lỗi

| Tình huống | Hành vi |
|---|---|
| Note thiếu `title` | Skip + log, không dừng cả run |
| Ảnh embed không tìm thấy | Giữ text, log cảnh báo |
| Vault path không tồn tại | Dừng ngay với thông báo rõ |
| `git push` lỗi | Dừng, giữ commit local, in hướng dẫn |
| Không có thay đổi | Bỏ qua commit, in "nothing to sync" |

## 7. Kiểm thử

- **Unit (transformer):** phân giải wikilink (đích public/không public/không có `|`), map frontmatter, tạo slug (có dấu tiếng Việt), stamp date theo thứ tự ưu tiên.
- **Unit (manifest):** thêm mới / cập nhật / gỡ bài.
- **Integration:** chạy `--dry-run` trên vault thật, kiểm kế hoạch in ra hợp lý.

## 8. Cấu trúc file (trong repo site)

```
mastersunflowers.github.io/
├── sync.py                 # entry point + git
├── sync_config.yml
├── wiki_sync/
│   ├── scanner.py
│   ├── transformer.py
│   ├── writer.py
│   └── manifest.py
├── tests/
│   └── test_transformer.py
├── .wiki-sync.json         # manifest (commit cùng repo)
└── requirements.txt        # python-frontmatter, PyYAML
```

## 9. Ngoài phạm vi (YAGNI)

- Tự động chạy theo lịch (cron/Task Scheduler) — có thể thêm sau bằng cách gọi `sync.py`.
- Render nâng cao callout/Obsidian-specific syntax.
- Đồng bộ ngược (site → vault).
