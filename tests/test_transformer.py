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
