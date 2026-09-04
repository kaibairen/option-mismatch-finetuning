from option_mismatch.behavior import analyze_behavior, extract_letter, parse_option_value


OPTIONS = ["A)125", "B)150", "C)225", "D)250", "E)275"]


def test_extract_letter_prefers_final_answer():
    text = "Maybe B.\nFinal answer: A"
    assert extract_letter(text) == "A"


def test_parse_option_value():
    assert parse_option_value("A)125") == 125.0
    assert parse_option_value("C) 20%") == 20.0


def test_mismatch_when_phase2_number_matches_other_option():
    generation = (
        "Profit per bag = 0.25. Total profit = 500 * 0.25 = 125.\n"
        "Final answer: C"
    )
    result = analyze_behavior(generation, OPTIONS, gold="A")
    assert result["inferred_letter"] == "A"
    assert result["number_match"] is True
    assert result["letter_correct"] is False
    assert result["mismatch"] is True
    assert result["format_ok"] is True


def test_no_mismatch_when_letter_and_number_agree():
    generation = "Total profit = 125.\nFinal answer: A"
    result = analyze_behavior(generation, OPTIONS, gold="A")
    assert result["letter_correct"] is True
    assert result["mismatch"] is False
    assert result["number_match"] is True


def test_wrong_without_phase2_number_is_not_mismatch():
    generation = "I guess.\nFinal answer: B"
    result = analyze_behavior(generation, OPTIONS, gold="A")
    assert result["letter_correct"] is False
    assert result["mismatch"] is False
