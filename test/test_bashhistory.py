import ueba.parse.bashhistory as to_test
import string


line1 = "the quick brown fox jumped"
line2 = "over the lazy dog"
words_line1 = line1.split()
words_line2 = line2.split()
unique_words = set(words_line1).union(set(words_line2))
num_unique_words = len(unique_words)
line2_bigrams = ["overthe", "thelazy", "lazydog"]


def test_delimiting():
    assert(to_test.splitting("ls -ltr /home/henryp/test.txt".split()) == "home henryp test.txt".split())


def test_max_length_of():
    assert(to_test.max_length_of([words_line1, words_line2]) == 5)


def test_bigrams():
    assert(to_test.ngrams_of(words_line2, 2) == line2_bigrams)


def test_trigrams():
    assert(to_test.all_ngrams_of(words_line2, 3) == line2_bigrams + ['overthelazy', 'thelazydog'])


def test_enhance():
    assert(to_test.enhance([line2], 2) == ["{} {}".format(line2, " ".join(line2_bigrams))])


def test_index_lines():
    word_to_index = to_test.index_words([line1, line2])
    assert(len(word_to_index) == num_unique_words)
    indices = word_to_index.values()
    assert(len(indices) == num_unique_words)
    assert(min(indices) == 0)
    assert(max(indices) == num_unique_words - 1)


def test_vectorize():
    word_to_index = to_test.index_words([line1, line2])
    index_to_word = dict((v, k) for k, v in word_to_index.items())
    vectors = to_test.vectorize([line1, line2], word_to_index)
    assert(len(vectors) == 2)
    actual = set()
    for vector in vectors:
        print("vector = {}, dictionary = {}".format(vector, index_to_word))
        words = [index_to_word[i] for i in vector]
        actual.add(" ".join(words))
    assert(actual == set([line1, line2]))


def test_truncate_or_pad():
    vectors = [[1, 2, 3], [4, 5, 6, 7, 8]]
    actual = to_test.truncate_or_pad(vectors, 4)
    assert(actual == [[1, 2, 3, 0], [4, 5, 6, 7]])
