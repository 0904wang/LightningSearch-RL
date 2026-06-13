from lightningsearch_rl.query import build_search_query


def test_build_search_query_expands_birthplace_to_born_city():
    query = build_search_query("Which city is the birthplace of the author of Example Book?")

    assert query.endswith("born city")
