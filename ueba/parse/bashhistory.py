import sys


def index_words(lines):
    max_index = 0
    word_index = {}
    for line in lines:
        xs = line.split()
        for x in xs:
            if word_index.get(x) is None:
                word_index[x] = max_index
                max_index += 1
    return word_index


if __name__ == "__main__":
    history1 = sys.argv[1]

    print(index_words(open(history1, "r")))
