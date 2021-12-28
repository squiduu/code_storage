import collections
import unicodedata


def load_vocab(vocab_file):
    """load subwords from .txt vocab file"""

    # set dictionary variable in the order of (subword, id)
    vocab = collections.OrderedDict()
    # set dictionary variable in the order of (id, subword)
    ids_to_tokens = collections.OrderedDict()

    # set index
    index = 0
    # open vocab file
    with open(vocab_file, "r", encoding="utf-8") as f:
        while True:
            # get subwords line by line
            token = f.readline()
            # in case of over the index
            if not token:
                break
            # remove meaningless whitespaces
            token = token.strip()

            # make dictionary variable of (subword, id)
            vocab[token] = index
            # make dictionary variable of (id, subword)
            ids_to_tokens[index] = token
            index += 1

    return vocab, ids_to_tokens


def _is_punctuation(char):
    """checks whether 'chars' equals a punctuation character"""

    # gets ascii code value for a specific character
    cp = ord(char)

    # returns True in case of special characters
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True

    # gets unicode category
    cat = unicodedata.category(char)
    # returns True in case of punctuation
    if cat.startswith("P"):
        return True
    return False


def _is_control(char):
    """checks whether 'chars' equals a control character"""

    if char == "\t" or char == "\n" or char == "\r":
        return False

    # gets unicode category
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_whitespace(char):
    """checks whether `chars` is a whitespace character"""

    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True

    # gets unicode category
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def whitespace_tokenize(text):
    """runs basic whitespaces cleaning and splitting on a piece of text"""

    text = text.strip()
    if not text:
        return []
    tokens = text.split()

    return tokens


class BasicTokenizer:
    """
    basic tokenizer for punctuation splitting, lower casing, etc.
    this is from HuggingFace
    """

    def __init__(
        self,
        do_lower_case=True,
        never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"),
    ):
        super().__init__()
        """
        Args:
            do_lower_case: whether to make lower case the input
            never_split: words that do not divide and is regarded as one
        """

        self.do_lower_case = do_lower_case
        self.never_split = never_split

    def tokenize(self, text):
        """tokenizes a piece of text"""

        text = self._clean_text(text)
        text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)

        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case and token not in self.never_split:
                token = token.lower()
                token = self._strip_accents(token)
            split_tokens.extend(self._split_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))

        return output_tokens

    def _clean_text(self, text):
        """remove invalid characters and meaningless whitespaces"""

        # sets initialized output list
        output = []
        # for each char
        for char in text:
            # gets ascii code value for a specific character
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)

        return "".join(output)

    def _tokenize_chinese_chars(self, text):
        """adds whitespace around any CJK character"""

        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)

        return "".join(output)

    def _is_chinese_char(self, cp):
        """checks whether CP is  codepoint of a CJK character"""

        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)
            or (cp >= 0x20000 and cp <= 0x2A6DF)
            or (cp >= 0x2A700 and cp <= 0x2B73F)
            or (cp >= 0x2B740 and cp <= 0x2B81F)
            or (cp >= 0x2B820 and cp <= 0x2CEAF)
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)
        ):

            return True

        return False

    def _strip_accents(self, text):
        """strip accents from a piece of text"""

        # normalize text with NFD method
        # normalizing is necessary because unicode may be different even with same characters
        text = unicodedata.normalize("NFD", text)

        # set output list
        output = []
        # for each character
        for char in text:
            # get unicode category of each char
            cat = unicodedata.category(char)
            # 'Mn' is a category of accents
            if cat == "Mn":
                continue
            # output appends only non-accents characters
            output.append(char)

        return "".join(output)

    def _split_punc(self, text):
        """splits punctuation on a piece of text"""

        # in case of text equals special tokens
        if text in self.never_split:
            return [text]

        chars = list(text)
        # sets initialized index
        i = 0
        start_new_word = True
        # sets initialized output list
        output = []
        while i < len(chars):
            # sets char
            char = chars[i]
            if _is_punctuation(char):
                # appends only in case of punctuation
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                # appends char at the end of output
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]


class WordPieceTokenizer:
    """runs word-piece tokenization"""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """tokenizes a piece of text into its word pieces
        For example:
            input = "unaffable"
            output = ["un", "##aff", "##able"]
        Arguments:
            text: a single token or whitespace separated tokens have been passed BasicTokenizer
        Returns:
            a list of word-piece tokens
        """

        # sets initialized output tokens
        output_tokens = []

        # for each cleaned text
        for token in whitespace_tokenize(text):
            # makes a list that is seratated into alphabets
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                # does not tokenize if char is too long
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                current_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        current_substr = substr
                        break
                    end -= 1
                if current_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(current_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)

        return output_tokens


class BertTokenizer:
    """word-piece tokenizing class for BERT"""

    def __init__(self, vocab_file, do_lower_case=True):
        """
        Args:
            vocab_file: path for vocabulary file
            do_lower_case: whether to make lower case in preprocessing
        """

        # loads vocabulary
        self.vocab, self.ids_to_tokens = load_vocab(vocab_file)
        # sets words that do not divide and is regarded as one
        never_split = ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")
        # sets tokenizing class
        self.basic_tokenizer = BasicTokenizer(
            do_lower_case=do_lower_case, never_split=never_split
        )
        self.wordpiece_tokenizer = WordPieceTokenizer(vocab=self.vocab)

    def tokenize(self, text):
        """seperates text to subwords"""

        # sets initialized split tokens list
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)

        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        """converts seperated token list to subword ID"""

        ids = []
        for token in tokens:
            ids.append(self.vocab[token])

        return ids

    def convert_ids_to_tokens(self, ids):
        """converts subword ID to seperated token list"""

        tokens = []
        for id in ids:
            tokens.append(self.ids_to_tokens[id])

        return tokens
