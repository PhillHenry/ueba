import ueba.parse.bashhistory as to_test


def test_index_lines():
    line1 = "the quick brown fox jumped"
    line2 = "over the lazy dog"
    word_to_index = to_test.index_words([line1, line2])
    words_line1 = line1.split()
    words_line2 = line2.split()
    unique_words = set(words_line1).union(set(words_line2))
    num_unique_words = len(unique_words)
    assert(len(word_to_index) == num_unique_words)
    indices = word_to_index.values()
    assert(len(indices) == num_unique_words)
    assert(min(indices) == 0)
    assert(max(indices) == num_unique_words - 1)
