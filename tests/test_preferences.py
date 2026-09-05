from option_mismatch.preferences import chosen_text


def test_chosen_appends_final_answer():
    text = chosen_text({"rationale": "Total = 125", "correct": "A"})
    assert "Total = 125" in text
    assert "Final answer: A" in text
