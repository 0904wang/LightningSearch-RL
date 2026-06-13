from lightningsearch_rl.actions import parse_action


def test_parse_search_action_extracts_query():
    action = parse_action("<search>Example Book author</search>")

    assert action.type == "search"
    assert action.content == "Example Book author"
    assert action.valid is True


def test_parse_answer_action_extracts_answer():
    action = parse_action("<answer>Example City</answer>")

    assert action.type == "answer"
    assert action.content == "Example City"
    assert action.valid is True


def test_parse_empty_search_is_invalid():
    action = parse_action("<search>   </search>")

    assert action.type == "search"
    assert action.valid is False
