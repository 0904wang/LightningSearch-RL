from lightningsearch_rl.data import Passage
from lightningsearch_rl.index_store import load_lexical_index, save_lexical_index


def test_save_and_load_lexical_index_preserves_search(tmp_path):
    index_path = tmp_path / "index.json"
    passages = [
        Passage("doc1", "Alpha", "Alpha text."),
        Passage("doc2", "Beta", "Beta born city."),
    ]

    save_lexical_index(index_path, passages)
    retriever = load_lexical_index(index_path)

    assert retriever.search("born city", top_k=1)[0].doc_id == "doc2"
