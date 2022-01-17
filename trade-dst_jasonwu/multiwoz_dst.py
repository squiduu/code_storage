from tqdm import tqdm
from config import *
from torch.utils.data import DataLoader
import torch.utils as utils
import os
import json
from fix_label import fix_label_error
import pickle
from embeddings import GloveEmbedding, KazumaCharEmbedding

# experiment only these domains due to lack of dialogues
experiment_domains = ["hotel", "train", "restaurant", "attraction", "taxi"]


class CustomVocab:
    def __init__(self):
        self.ids_to_words = {
            UNK_TOKEN: "UNK",
            PAD_TOKEN: "PAD",
            EOS_TOKEN: "EOS",
            SOS_TOKEN: "SOS",
        }
        # get the number of special tokens
        self.n_words = len(self.ids_to_words)
        self.words_to_ids = {}
        for k, v in self.ids_to_words.items():
            self.words_to_ids[v] = k

    def add_word(self, word):
        """add new special tokens"""
        if word not in self.words_to_ids:
            self.words_to_ids[word] = self.n_words
            self.ids_to_words[self.n_words] = word
            self.n_words += 1

    def add_words(self, sent, type):
        if type == "uttr":
            for word in sent.split(" "):
                self.add_word(word)
        elif type == "slot":
            for slots in sent:
                domain, slot = slots.split("-")
                self.add_word(domain)
                for subslot in slot.split(" "):
                    self.add_word(subslot)
        elif type == "belief":
            for slots, values in sent.items():
                domain, slot = slots.split("-")
                self.add_word(domain)
                for subslot in slot.split(" "):
                    self.add_word(subslot)
                for value in values.split(" "):
                    self.add_word(value)


class CustomDataset(utils.data.Dataset):
    """read source and target sequences from .txt files"""

    def __init__(
        self, data_info: dict, src_words_to_ids, trg_words_to_ids, mem_words_to_ids,
    ):
        super().__init__()

        # get data infomation
        self.ID = data_info["ID"]
        self.turn_domain = data_info["turn_domain"]
        self.turn_id = data_info["turn_id"]
        self.dialog_history = data_info["dialog_history"]
        self.turn_belief = data_info["turn_belief"]
        self.gate_label = data_info["gate_label"]
        self.turn_uttr = data_info["turn_uttr"]
        self.gen_slotvalue = data_info["gen_slotvalue"]
        self.src_words_to_ids = src_words_to_ids
        self.trg_words_to_ids = trg_words_to_ids
        self.mem_words_to_ids = mem_words_to_ids

    def __len__(self):
        return len(self.dialog_history)

    def __getitem__(self, index):
        """returns a indexed single data pair of source and target"""
        id = self.ID[index]
        turn_domain = self.get_domain_number(self.turn_domain[index])
        turn_id = self.turn_id[index]
        context = self.dialog_history[index]
        context = self.convert_context_words_to_ids(
            sequence=context, words_to_ids=self.src_words_to_ids
        )
        context_plain = self.dialog_history[index]
        turn_belief = self.turn_belief[index]
        gate_label = self.gate_label[index]
        turn_uttr = self.turn_uttr[index]
        gen_slotvalue = self.gen_slotvalue[index]
        gen_slotvalue = self.convert_slot_words_to_ids(
            sequence=gen_slotvalue, words_to_ids=self.trg_words_to_ids
        )

        item_info = {
            "ID": id,
            "turn_domain": turn_domain,
            "turn_id": turn_id,
            "context": context,
            "context_plain": context_plain,
            "turn_belief": turn_belief,
            "gate_label": gate_label,
            "turn_uttr": turn_uttr,
            "gen_slotvalue": gen_slotvalue,
        }

        return item_info

    def get_domain_number(self, turn_domain):
        domains = {
            "attraction": 0,
            "restaurant": 1,
            "taxi": 2,
            "train": 3,
            "hotel": 4,
            "hospital": 5,
            "bus": 6,
            "police": 7,
        }

        return domains[turn_domain]

    def convert_context_words_to_ids(self, sequence: str, words_to_ids):
        story = []
        for word in sequence.split():
            if word in words_to_ids:
                story.append(words_to_ids[word])
            else:
                story.append(UNK_TOKEN)

        story = torch.Tensor(story)

        return story

    def convert_slot_words_to_ids(self, sequence: str, words_to_ids):
        story = []
        for value in sequence:
            story_temp = []
            for word in value.split():
                if word in words_to_ids:
                    story_temp.append(words_to_ids[word])
                else:
                    story_temp.append(UNK_TOKEN)
            story_temp.append(EOS_TOKEN)
            story.append(story_temp)

        return story


def read_and_organize_dataset(
    file: str,
    slot_gate: dict,
    slots: list,
    mode: str,
    vocab: CustomVocab,
    mem_vocab: CustomVocab,
    training: bool,
    max_line=None,
):
    """read dataset and organize to paired for training"""
    print(f"READING FROM {file}")

    # preset
    data_list = []
    max_resp_len = 0
    max_value_len = 0
    domain_counters = {}

    # load dialogues data
    with open(file) as f:
        dialogs = json.load(f)
        # add vocab from transcripts
        for dialog in dialogs:
            if (args.b_all_vocab or mode == "train") and training:
                for ti, turn in enumerate(dialog["dialogue"]):
                    vocab.add_words(sent=turn["system_transcript"], type="uttr")
                    vocab.add_words(sent=turn["transcript"], type="uttr")

        line_counter = 1
        for dialog in dialogs:
            dialog_history = ""
            # filter and count domains
            for domain in dialog["domains"]:
                if domain not in experiment_domains:
                    continue
                if domain not in domain_counters.keys():
                    domain_counters[domain] = 0
                domain_counters[domain] += 1

            # read and organize data
            for ti, turn in enumerate(dialog["dialogue"]):
                turn_domain = turn["domain"]
                turn_id = turn["turn_idx"]
                turn_uttr = turn["system_transcript"] + " ; " + turn["transcript"]
                turn_uttr = turn_uttr.strip()
                dialog_history += (
                    turn["system_transcript"] + " ; " + turn["transcript"] + " ; "
                )
                dialog_history = dialog_history.strip()
                turn_belief_dict = fix_label_error(
                    labels=turn["belief_state"], type=False, slots=slots
                )
                turn_belief_list = []
                for k, v in turn_belief_dict.items():
                    turn_belief_list.append(str(k) + "-" + str(v))

                # add vocab from belief state
                if (args.b_all_vocab or mode == "train") and training:
                    mem_vocab.add_words(sent=turn_belief_dict, type="belief")

                gen_slotvalue = []
                gate_label = []
                for slot in slots:
                    if slot in turn_belief_dict.keys():
                        gen_slotvalue.append(turn_belief_dict[slot])
                        # get gate label list
                        if turn_belief_dict[slot] == "dontcare":
                            gate_label.append(slot_gate["dontcare"])
                        elif turn_belief_dict[slot] == "none":
                            gate_label.append(slot_gate["none"])
                        else:
                            gate_label.append(slot_gate["ptr"])

                        # match length
                        if max_value_len < len(turn_belief_dict[slot]):
                            max_value_len = len(turn_belief_dict[slot])

                    else:
                        gen_slotvalue.append("none")
                        gate_label.append(slot_gate["none"])

                # organize data from dialogues
                detailed_data = {
                    "ID": dialog["dialogue_idx"],
                    "domains": dialog["domains"],
                    "turn_domain": turn_domain,
                    "turn_id": turn_id,
                    "dialog_history": dialog_history,
                    "turn_belief": turn_belief_list,
                    "gate_label": gate_label,
                    "turn_uttr": turn_uttr,
                    "gen_slotvalue": gen_slotvalue,
                }
                data_list.append(detailed_data)

                # get max dialogue history length
                if max_resp_len < len(dialog_history.split()):
                    max_resp_len = len(dialog_history.split())

            # get the number of lines read
            line_counter += 1
            if max_line and line_counter >= max_line:
                break

    # add vocab with f"t{i}" itself
    if f"t{max_value_len - 1}" not in mem_vocab.words_to_ids.keys() and training:
        for i in range(max_value_len):
            mem_vocab.add_words(sent=f"t{i}", type="uttr")

    print("domain counter:", domain_counters)

    return data_list, max_resp_len, slots


def get_slot_info(ontology):
    """get domain-slot names as list"""
    ontology_domains = []
    for k, v in ontology.items():
        if k.split("-")[0] in experiment_domains:
            ontology_domains.append((k, v))
    ontology_domains = dict(ontology_domains)

    slots = []
    for k in ontology_domains.keys():
        if "book" not in k:
            slot = k.replace(" ", "")
            slot = slot.lower()
            slots.append(slot)
        else:
            slot = k.lower()
            slots.append(slot)

    return slots


def collate_fn(data_list: list):
    """match the tensor size if the input size varies by batch"""

    def merge(sequences):
        """(batch_size, sent_len) -> (batch_size, max_seq_len)"""
        # make a list of sequences length
        lengths = []
        for sequence in sequences:
            lengths.append(len(sequence))

        # get max_seq_len
        if max(lengths) == 0:
            max_seq_len = 1
        else:
            max_seq_len = max(lengths)

        # make a padded tensor
        padded_seqs = torch.ones(len(sequences), max_seq_len).long()
        for i, seq in enumerate(sequences):
            padded_seqs[i, : lengths[i]] = seq[: lengths[i]]
        # copy a tensor with required_grad equals as False
        padded_seqs = padded_seqs.detach()

        return padded_seqs, lengths

    def merge_multi_responses(sequences):
        """(batch_size, n_slots, slot_len) -> (batch_size, n_slots, max_slot_len)"""
        lengths = []
        for sequence in sequences:
            length = []
            for seq in sequence:
                length.append(len(seq))
            lengths.append(length)

        max_len = []
        for length in lengths:
            max_len.append(len(length))
        max_seq_len = max(max_len)

        padded_seqs = []
        for sequence in sequences:
            pad_seq = []
            for seq in sequence:
                seq = seq + [PAD_TOKEN] * (max_seq_len - len(seq))
                pad_seq.append(seq)
            padded_seqs.append(pad_seq)
        padded_seqs = torch.tensor(padded_seqs)
        lengths = torch.tensor(lengths)

        return padded_seqs, lengths

    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data_list.sort(key=lambda x: len(x["context"]), reverse=True)
    item_info = {}
    for key in data_list[0].keys():
        item_info[key] = [data[key] for data in data_list]

    # merge sequences
    src_seqs, src_lengths = merge(item_info["context"])
    y_seqs, y_lengths = merge_multi_responses(item_info["gen_slotvalue"])
    gate_label = torch.tensor(item_info["gate_label"])
    turn_domain = torch.tensor(item_info["turn_domain"])

    if USE_CUDA:
        src_seqs = src_seqs.to(device)
        gate_label = gate_label.to(device)
        turn_domain = turn_domain.to(device)
        slotvalue_seqs = y_seqs.to(device)
        slotvalue_lengths = y_lengths.to(device)

    item_info["context"] = src_seqs
    item_info["context_len"] = src_lengths
    item_info["gate_label"] = gate_label
    item_info["turn_domain"] = turn_domain
    item_info["gen_slotvalue"] = slotvalue_seqs
    item_info["slotvalue_len"] = slotvalue_lengths

    return item_info


def get_dataloader(
    pairs: list, vocab: CustomVocab, mem_vocab: CustomVocab, batch_size, shuffle
):
    # make empty list with shape of detailed data dict
    data_info = {}
    for k in pairs[0].keys():
        data_info[k] = []
    # fill data info list
    for pair in pairs:
        for k in pairs[0].keys():
            data_info[k].append(pair[k])

    # get organized data
    custom_ds = CustomDataset(
        data_info=data_info,
        src_words_to_ids=vocab.words_to_ids,
        trg_words_to_ids=vocab.words_to_ids,
        mem_words_to_ids=mem_vocab.words_to_ids,
    )
    # get dataloader
    custom_dl = DataLoader(
        dataset=custom_ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
    )

    return custom_dl


def dump_pretrained_vocab_ids_to_words(words_to_ids, ids_to_words, dump_path):
    print("DUMP PRE-TRAINED EMBEDDINGS")
    embedding_list = [GloveEmbedding(), KazumaCharEmbedding()]
    embs = []
    for i in tqdm(range(len(words_to_ids.keys()))):
        word = ids_to_words[i]
        emb = []
        for embedding in embedding_list:
            emb += embedding.emb(word, default="zero")
        embs.append(emb)
    with open(dump_path, "wt") as f:
        json.dump(embs, f)


def prepare_dataloader(training, task="dst", batch_size=100):
    # batch_size for evaluation
    if args.eval_batch_size:
        eval_batch_size = args.eval_batch_size
    else:
        eval_batch_size = batch_size

    # set datasets
    train_file = "./data/train_dials.json"
    valid_file = "./data/valid_dials.json"
    test_file = "./data/test_dials.json"

    # set path to save model
    if args.path:
        folder = args.path.rsplit("/", 2)[0] + "/"
    else:
        folder = f"save/{args.decoder}_{args.add_name}_{args.dataset}_{args.task}/"
    print("SAVE FOLDER:", folder)
    # make a new folder if does not exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    # load slot-value pairs from ontology
    with open("./data/multi-woz/MULTIWOZ2 2/ontology.json", "r") as f:
        ontology = json.load(f)
    slots = get_slot_info(ontology)

    # set slot gate dict
    slot_gate = {"ptr": 0, "dontcare": 1, "none": 2}

    # set vocab
    vocab = CustomVocab()
    mem_vocab = CustomVocab()
    # add slot vocab
    vocab.add_words(sent=slots, type="slot")
    if args.b_all_vocab:
        vocab_name = "vocab_all.pkl"
    else:
        vocab_name = "vocab_train.pkl"
    # add slot vocab
    mem_vocab.add_words(sent=slots, type="slot")
    if args.b_all_vocab:
        mem_vocab_name = "mem_vocab_all.pkl"
    else:
        mem_vocab_name = "mem_vocab_train.pkl"

    # make dataloader and vocab file in training mode
    if training:
        # get training dataloader
        pair_train, train_max_len, slot_train = read_and_organize_dataset(
            file=train_file,
            slot_gate=slot_gate,
            slots=slots,
            mode="train",
            vocab=vocab,
            mem_vocab=mem_vocab,
            training=True,
        )
        train_dl = get_dataloader(
            pairs=pair_train,
            vocab=vocab,
            mem_vocab=mem_vocab,
            batch_size=batch_size,
            shuffle=True,
        )
        n_train_vocab = vocab.n_words

        # get validation dataloader
        pair_valid, valid_max_len, slot_valid = read_and_organize_dataset(
            file=valid_file,
            slot_gate=slot_gate,
            slots=slots,
            mode="valid",
            vocab=vocab,
            mem_vocab=mem_vocab,
            training=training,
        )
        valid_dl = get_dataloader(
            pairs=pair_valid,
            vocab=vocab,
            mem_vocab=mem_vocab,
            batch_size=batch_size,
            shuffle=False,
        )

        # get test dataloader
        pair_test, test_max_len, slot_test = read_and_organize_dataset(
            file=test_file,
            slot_gate=slot_gate,
            slots=slots,
            mode="test",
            vocab=vocab,
            mem_vocab=mem_vocab,
            training=training,
        )
        test_dl = get_dataloader(
            pairs=pair_test,
            vocab=vocab,
            mem_vocab=mem_vocab,
            batch_size=batch_size,
            shuffle=False,
        )

        # load vocab files if it exists
        if os.path.exists(folder + vocab_name) and os.path.exists(
            folder + mem_vocab_name
        ):
            print(f"LOAD SAVED VOCAB FILES")
            with open(folder + vocab_name, "rb") as f:
                vocab = pickle.load(f)
            with open(folder + mem_vocab_name, "rb") as f:
                mem_vocab = pickle.load(f)
        # save vocab files if it does not exist
        else:
            print(f"DUMP VOCAB FILES")
            with open(folder + vocab_name, "wb") as f:
                pickle.dump(obj=vocab, file=f)
            with open(folder + mem_vocab_name, "wb") as f:
                pickle.dump(obj=mem_vocab, file=f)

        # save vocab.ids_to_words file
        vocab_ids_to_words_path = f"./data/emb_{len(vocab.ids_to_words)}.json"
        if not os.path.exists(vocab_ids_to_words_path) and args.b_load_embedding:
            dump_pretrained_vocab_ids_to_words(
                words_to_ids=vocab.words_to_ids,
                ids_to_words=vocab.ids_to_words,
                dump_path=vocab_ids_to_words_path,
            )

    # load vocab in validation or testing mode
    else:
        # load pre-trained vocab files
        with open(folder + vocab_name, "rb") as f:
            vocab = pickle.load(f)
        with open(folder + mem_vocab_name, "rb") as f:
            mem_vocab = pickle.load(f)

        # set empty objects in case of non-training mode
        pair_train = []
        train_max_len = 0
        slot_train = {}
        train_dl = []
        n_train_vocab = 0

        # get validation dataloader
        pair_valid, valid_max_len, slot_valid = read_and_organize_dataset(
            file=valid_file,
            slot_gate=slot_gate,
            slots=slots,
            mode="valid",
            vocab=vocab,
            mem_vocab=mem_vocab,
            training=training,
        )
        valid_dl = get_dataloader(
            pairs=pair_valid,
            vocab=vocab,
            mem_vocab=mem_vocab,
            batch_size=eval_batch_size,
            shuffle=False,
        )

        # get test dataloader
        pair_test, test_max_len, slot_test = read_and_organize_dataset(
            file=test_file,
            slot_gate=slot_gate,
            slots=slots,
            mode="test",
            vocab=vocab,
            mem_vocab=mem_vocab,
            training=training,
        )
        test_dl = get_dataloader(
            pairs=pair_test,
            vocab=vocab,
            mem_vocab=mem_vocab,
            batch_size=eval_batch_size,
            shuffle=False,
        )

    # get max words len in dataset
    max_word_len = max(train_max_len, valid_max_len, test_max_len) + 1

    # notify dataloader status
    print(f"Read {len(pair_train)}-pairs training data")
    print(f"Read {len(pair_valid)}-pairs validation data")
    print(f"Read {len(pair_test)}-pairs testing data")
    print(f"Vocab size: {vocab.n_words}")
    print(f"Vocab size for training: {n_train_vocab}")
    print(f"Vocab size for belief states: {mem_vocab.n_words}")
    print(f"Max lenght of dialogue words for RNN: {max_word_len}")
    print(f"USE CUDA: {USE_CUDA} & {device}")

    total_slots = [slots, slot_train, slot_valid, slot_test]
    print(f"Train and valid dataset slots are {str(len(total_slots[1:2]))} in total")
    print(total_slots[1:2])
    print(f"Test slots are {str(len(total_slots[3]))} in total")
    print(total_slots[3])

    total_vocab = [vocab, mem_vocab]

    return (
        train_dl,
        valid_dl,
        test_dl,
        total_vocab,
        total_slots,
        slot_gate,
        n_train_vocab,
    )
