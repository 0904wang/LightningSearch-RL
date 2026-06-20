from lightningsearch_rl.agent_loop import SearchEnvironment, parse_agent_action
from lightningsearch_rl.data import Passage
from lightningsearch_rl.retrieval import LexicalRetriever


def test_parse_search_action_allows_one_clean_search():
    action = parse_agent_action("<search>Dr. Elena Voss grant organization</search>")

    assert action.type == "search"
    assert action.query == "Dr. Elena Voss grant organization"
    assert action.answer is None
    assert action.valid is True


def test_parse_answer_action_allows_one_clean_answer():
    action = parse_agent_action("<answer>National Science Foundation</answer>")

    assert action.type == "answer"
    assert action.answer == "National Science Foundation"
    assert action.query is None
    assert action.valid is True


def test_parse_action_rejects_model_generated_observation():
    action = parse_agent_action("<observation>[1] fabricated evidence</observation>")

    assert action.type == "invalid"
    assert action.valid is False
    assert action.reason == "model_generated_observation"


def test_parse_action_rejects_multiple_actions():
    action = parse_agent_action("<search>query</search><answer>Answer</answer>")

    assert action.type == "invalid"
    assert action.valid is False
    assert action.reason == "multiple_actions"


def test_parse_action_rejects_answer_inside_search():
    action = parse_agent_action("<search>Who won? <answer>Prize</answer></search>")

    assert action.type == "invalid"
    assert action.valid is False
    assert action.reason == "nested_answer_in_search"


def test_search_environment_inserts_formatted_observation():
    retriever = LexicalRetriever(
        [
            Passage(
                doc_id="doc1",
                title="Grant Report",
                text="The research center was funded by the National Science Foundation.",
            )
        ]
    )
    env = SearchEnvironment(retriever, top_k=1)

    observation = env.search_observation("research center funded")

    assert observation.startswith("<observation>")
    assert "[1] Grant Report: The research center was funded by the National Science Foundation." in observation
    assert observation.endswith("</observation>")
