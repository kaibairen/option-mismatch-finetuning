from option_mismatch.span import find_phase3_char_start, resolve_phase3_char_start, split_ndi


def test_final_answer_cut():
    text = "Profit is 125.\nFinal answer: A"
    start, rule = find_phase3_char_start(text)
    assert rule == "final_answer"
    assert text[start:].startswith("Final answer")
    assert "Profit is 125" in text[:start]


def test_chinese_cut():
    text = "中间结果 40\n所以选 B"
    start, rule = find_phase3_char_start(text)
    assert rule == "so_choose_zh"
    assert "中间结果" in text[:start]


def test_option_align_fallback():
    text = "selling price = 50 so option C is the match"
    start, rule = resolve_phase3_char_start(text)
    assert rule == "option_align"
    assert start < len(text)


def test_split_ndi_respects_cut():
    ndi = [0.2, 0.2, 0.2, 0.0, 0.0]
    p2, p3 = split_ndi(ndi, token_start=3)
    assert abs(p2 - 0.2) < 1e-9
    assert abs(p3 - 0.0) < 1e-9
