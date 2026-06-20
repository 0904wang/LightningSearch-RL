from lightningsearch_rl.answer_metrics import score_answer
from lightningsearch_rl.answer_metrics import soft_answer_reward


def test_score_answer_reports_exact_f1_and_containment():
    exact = score_answer("Nobel Peace Prize", "nobel peace prize")
    suffix = score_answer("Golden Quill Award", "Golden Quill")
    wrong = score_answer("Global Health Research Institute", "Barcelona")

    assert exact["exact_match"] is True
    assert exact["token_f1"] == 1.0
    assert exact["containment_match"] is True
    assert suffix["exact_match"] is False
    assert suffix["token_f1"] == 0.8
    assert suffix["containment_match"] is True
    assert wrong["exact_match"] is False
    assert wrong["token_f1"] == 0.0
    assert wrong["containment_match"] is False


def test_soft_answer_reward_scores_containment_and_high_f1_partial_matches():
    exact = soft_answer_reward("Nobel Peace Prize", "nobel peace prize")
    award_suffix = soft_answer_reward("Golden Quill Award", "Golden Quill")
    venue_city = soft_answer_reward("Vienna Conference Center", "Vienna")
    high_f1 = soft_answer_reward("Global Health Research Institute", "Global Health Institute")
    wrong = soft_answer_reward("Global Health Research Institute", "Barcelona")

    assert exact["answer_reward"] == 1.0
    assert exact["answer_reward_type"] == "exact"
    assert award_suffix["answer_reward"] == 0.8
    assert award_suffix["answer_reward_type"] == "containment"
    assert venue_city["answer_reward"] == 0.5
    assert venue_city["answer_reward_type"] == "containment"
    assert high_f1["answer_reward"] == 0.857143
    assert high_f1["answer_reward_type"] == "token_f1"
    assert wrong["answer_reward"] == 0.0
    assert wrong["answer_reward_type"] == "none"
