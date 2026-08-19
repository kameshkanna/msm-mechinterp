from __future__ import annotations

from msm_mechinterp.data.prompts import PRO_AMERICA_VS_PRO_AFFORDABILITY, prompt_pairs_to_texts


def test_seed_pairs_are_nonempty_and_distinct() -> None:
    assert len(PRO_AMERICA_VS_PRO_AFFORDABILITY) > 0
    for pair in PRO_AMERICA_VS_PRO_AFFORDABILITY:
        assert pair.agenda_a_text.strip()
        assert pair.agenda_b_text.strip()
        assert pair.agenda_a_text != pair.agenda_b_text


def test_prompt_pairs_to_texts_splits_correctly() -> None:
    agenda_a_texts, agenda_b_texts = prompt_pairs_to_texts(PRO_AMERICA_VS_PRO_AFFORDABILITY)
    assert len(agenda_a_texts) == len(agenda_b_texts) == len(PRO_AMERICA_VS_PRO_AFFORDABILITY)
    assert agenda_a_texts[0] == PRO_AMERICA_VS_PRO_AFFORDABILITY[0].agenda_a_text
