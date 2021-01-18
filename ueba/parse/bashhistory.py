import sys


def index_words(lines, word_index={}):
    max_index = 0
    for line in lines:
        xs = line.split()
        for x in xs:
            if word_index.get(x) is None:
                word_index[x] = max_index
                max_index += 1
    return word_index


def vectorise(lines, word_index):
    vectors = []
    for line in lines:
        xs = line.split()
        vectors.append([word_index[x] for x in xs])
    return vectors



if __name__ == "__main__":
    dictionary = index_words(open(sys.argv[1], "r"))
    dictionary = index_words(open(sys.argv[2], "r"), dictionary)
