from glob import glob
import os


def make_tsv(file: str, path: str):
    file = open(file, "w")

    path_pos = path + "pos/"
    for fname in glob(os.path.join(path_pos, "*.txt")):
        with open(fname, "r", encoding="utf-8") as f:
            line = f.readline()
            line = line.replace("\t", " ").replace("<br />", "")
            line = line + "\t" + "1" + "\t" + "\n"

            file.write(line)

    path_neg = path + "neg/"
    for fname in glob(os.path.join(path_neg, "*.txt")):
        with open(fname, "r", encoding="utf-8") as f:
            line = f.readline()
            line = line.replace("\t", " ").replace("<br />", "")
            line = line + "\t" + "0" + "\t" + "\n"

            file.write(line)

    file.close()


make_tsv(
    file="./database/IMDB/aclImdb/train/imdb_train.tsv",
    path="./database/IMDB/aclImdb/train/",
)

make_tsv(
    file="./database/IMDB/aclImdb/test/imdb_test.tsv",
    path="./database/IMDB/aclImdb/test/",
)
