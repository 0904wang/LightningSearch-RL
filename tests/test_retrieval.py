from lightningsearch_rl.data import Passage
from lightningsearch_rl.retrieval import LexicalRetriever


def test_search_ranks_matching_passage_first():
    retriever = LexicalRetriever(
        [
            Passage("doc_book", "Example Book", "Example Book was written by Alice Smith."),
            Passage("doc_author", "Alice Smith", "Alice Smith was born in Example City."),
            Passage("doc_noise", "Noise", "Unrelated text."),
        ]
    )

    results = retriever.search("Alice Smith born city", top_k=2)

    assert [result.doc_id for result in results] == ["doc_author", "doc_book"]
