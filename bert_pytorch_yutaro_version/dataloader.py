import re
import string
import random
from tokenizer import BertTokenizer, load_vocab
import torchtext


def preprocess_text(text):
    """preprocesses of IMDb text"""

    # removes newline characters
    text = re.sub("<br />", "", text)

    # replaces punctuations to whitespaces except . or ,
    for punc in string.punctuation:
        if punc == "." or punc == ",":
            continue
        else:
            text = text.replace(punc, " ")

    # adds whitespaces back and forth of punctuations
    text = text.replace(".", " . ")
    text = text.replace(",", " , ")

    return text


# sets BERT tokenizer
tokenizer = BertTokenizer("./vocab/bert-base-uncased-vocab.txt")


def preprocess_and_tokenize(text):
    """preprocesses and tokenizes of IMDb text"""

    # preprocesses text
    text = preprocess_text(text)
    # tokenizes text
    tokens = tokenizer.tokenize(text)

    return tokens


# sets max sequence length
max_length = 512
# sets data field for text and label
TEXT = torchtext.legacy.data.Field(
    # whether data length is variable
    sequential=True,
    # whether to use vocab object
    use_vocab=True,
    init_token="[CLS]",
    eos_token="[SEP]",
    fix_length=max_length,
    tokenize=preprocess_and_tokenize,
    # whether to include the number of subwords data
    include_lengths=True,
    batch_first=True,
    pad_token="[PAD]",
    unk_token="[UNK]",
)
LABEL = torchtext.legacy.data.Field(sequential=False, use_vocab=False)

# loads train, validation, and test datasets
train_val_ds, test_ds = torchtext.legacy.data.TabularDataset.splits(
    path="./data/",
    train="imdb_train.tsv",
    test="imdb_test.tsv",
    format="tsv",
    # sets fields as [("field name", field object)]
    fields=[("Text", TEXT), ("Label", LABEL)],
)
# splits train and validation datasets
train_ds, val_ds = train_val_ds.split(split_ratio=0.8, random_state=random.seed(1234))

# loads vocab
vocab_bert, ids_to_tokens_bert = load_vocab("./vocab/bert-base-uncased-vocab.txt")

# needs 'TEXT.vocab.stoi' to connect ID and subwords when making dataloader
# needs to run 'build_vocab()' to make vocabulary set to assign loaded vocab_bert as 'TEXT.vocab.stoi'
TEXT.build_vocab(train_ds, min_freq=1)
TEXT.vocab.stoi = vocab_bert

# sets batch size
batch_size = 64

# sets data iterators
train_dl = torchtext.legacy.data.Iterator(train_ds, batch_size, train=True)
val_dl = torchtext.legacy.data.Iterator(val_ds, batch_size, train=False, sort=False)
test_dl = torchtext.legacy.data.Iterator(test_ds, batch_size, train=False, sort=False)

# organizes into dictionary object
dl_dict = {"train": train_dl, "val": val_dl}
