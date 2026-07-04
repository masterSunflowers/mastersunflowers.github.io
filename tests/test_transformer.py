from wiki_sync.transformer import slugify, extract_category
from wiki_sync.transformer import (
    build_published_index,
    resolve_wikilinks,
    rewrite_image_embeds,
    render_post,
)
from wiki_sync.scanner import Note
from pathlib import Path


def test_slugify_strips_vietnamese_diacritics():
    assert slugify("Hồi quy tuyến tính") == "hoi-quy-tuyen-tinh"


def test_slugify_handles_symbols_and_spacing():
    assert slugify("Continuous Control - PPO, SAC, TD3") == "continuous-control-ppo-sac-td3"


def test_extract_category_reads_topic_label():
    md = {"topics": ["[[Knowledge/LLM Wiki/Topics/RL for Robotics|RL for Robotics]]"]}
    assert extract_category(md, "LLM Wiki") == "RL for Robotics"


def test_extract_category_falls_back_to_default():
    assert extract_category({}, "LLM Wiki") == "LLM Wiki"


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
