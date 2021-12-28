import string
import re
import torchtext
import random
from torchtext.vocab import Vectors


def get_dataloader_and_text(max_length=256, batch_size=24):
    def preprocess_text(text):
        """clean text"""
        # remove <br /> in text
        text = re.sub("<br />", "", text)

        # replace unnecessary chars to whitespace except . and ,
        for punc in string.punctuation:
            if punc == "." or punc == ",":
                continue
            else:
                text = text.replace(punc, " ")

        # add whitespace before and after . and ,
        text = text.replace(".", " . ")
        text = text.replace(",", " , ")

        return text

    def tokenize_punctuation(text):
        """divide whitespace"""
        # strip() removes only the first and last whitespaces and returns as string
        # split() seperates based on the whitespace and returns as list
        return text.strip().split()

    def preprocess_and_tokenize(text):
        """set preprocess and tokenize functions"""
        text = preprocess_text(text)
        ret = tokenize_punctuation(text)

        return ret

    max_length = 256
    # define the preprocessing method as an Field object
    TEXT = torchtext.legacy.data.Field(
        # date type is sequential or not
        sequential=True,
        # make vocab set or not
        use_vocab=True,
        init_token="<cls>",
        eos_token="<eos>",
        fix_length=max_length,
        tokenize=preprocess_and_tokenize,
        lower=True,
        include_lengths=True,
        # set the first dim as mini-batch dim or not
        batch_first=True,
    )
    LABEL = torchtext.legacy.data.Field(sequential=False, use_vocab=False)

    # make train, val, and test datasets with the defined preprocessing method
    train_val_ds, test_ds = torchtext.legacy.data.TabularDataset.splits(
        path="./database/IMDB/aclImdb/",
        train="train/imdb_train.tsv",
        test="test/imdb_test.tsv",
        format="tsv",
        # set field as [("field name", Field object)]
        fields=[("Text", TEXT), ("Label", LABEL)],
    )

    # split train and val, datasets
    train_ds, val_ds = train_val_ds.split(split_ratio=0.8, random_state=random.seed(42))

    # define a pre-trained vocab vector model as vectors
    eng_fasttext_vec = Vectors(name="./database/wiki-news-300d-1M.vec")

    # eng_fasttext_vec -> Vectors class object
    # eng_fasttext_vec.dim -> dimension for an word
    # eng_fasttext_vec.stoi -> {'str': int}
    # eng_fasttext_vec.vectors[0] -> word embedding vector for the first word

    # make vocab set of train dataset with pre-defined preprocessing, and initialize embedding vectors
    TEXT.build_vocab(train_ds, vectors=eng_fasttext_vec, min_freq=10)

    # TEXT.vocab.vectors.shape -> TensorSize([17691, 300])
    # TEXT.vocab.vectors -> embedding vectors
    # TEXT.vocab.stoi -> {'str': int}

    # set train, val, and test dataloader
    # dataloader saves words as word IDs due to efficient memory usage
    # neural network model takes out vector representations based on the pre-defined word IDs
    train_dl = torchtext.legacy.data.Iterator(
        dataset=train_ds, batch_size=batch_size, train=True
    )
    val_dl = torchtext.legacy.data.Iterator(
        dataset=val_ds, batch_size=batch_size, train=False, sort=False
    )
    test_dl = torchtext.legacy.data.Iterator(
        dataset=test_ds, batch_size=batch_size, train=False, sort=False
    )

    return train_dl, val_dl, test_dl, TEXT
