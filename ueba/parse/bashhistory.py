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


def vectorize(lines, word_index):
    vectors = []
    for line in lines:
        xs = line.split()
        vectors.append([word_index[x] for x in xs])
    return vectors


def truncate_or_pad(vs, n):
    sized = []
    for v in vs:
        length = len(v)
        if length > n:
            sized.append(v[:n])
        else:
            padding = [0] * (n - length)
            v = v + padding
            sized.append(v)
    return sized


if __name__ == "__main__":
    dictionary = index_words(open(sys.argv[1], "r"))
    dictionary = index_words(open(sys.argv[2], "r"), dictionary)
