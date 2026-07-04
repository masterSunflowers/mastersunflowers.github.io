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
