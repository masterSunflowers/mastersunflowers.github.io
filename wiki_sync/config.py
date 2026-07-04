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
