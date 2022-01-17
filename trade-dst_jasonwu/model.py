import random
import numpy as np
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from config import *
import torch.nn as nn
import os
import torch
import json
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from multiwoz_dst import CustomVocab
import torch.optim as optim
from masked_cross_entropy import *


class UtteranceEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, p_dropout, n_layers=1):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.dropout = nn.Dropout(p_dropout)
        self.n_layers = n_layers
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.d_model,
            padding_idx=PAD_TOKEN,
        )
        self.gru = nn.GRU(
            d_model, d_model, n_layers, dropout=p_dropout, bidirectional=True
        )

        # normalize embedding weights
        self.embedding.weight.data.normal_(mean=0, std=0.1)

        # load pre-trained embedding if is true
        if args.b_load_embedding:
            with open(os.path.join("./data/", f"emb_{self.vocab_size}.json")) as f:
                loaded_embedding = json.load(f)
            new = self.embedding.weight.data.new
            self.embedding.weight.data.copy_(new(loaded_embedding))
            # turn on grad
            self.embedding.weight.requires_grad = True

        # no grad setting (optional)
        if args.b_fix_embedding:
            self.embedding.weight.requires_grad = False

    def create_hidden(self, batch_size):
        """create empty cell states and hidden states"""
        if USE_CUDA:
            return torch.zeros(size=(2, batch_size, self.d_model)).to(device)
        else:
            return torch.zeros(size=(2, batch_size, self.d_model))

    def forward(self, inputs: torch.Tensor, input_len, hidden: torch.Tensor = None):
        """
        Args:
            inputs: recent dialogue histories
            input_len: length of input dialogue histories
        """
        # token embedding
        embedding = self.embedding(inputs)
        # token embedding applied dropout layer
        embedding = self.dropout(embedding)
        # create empty hidden states for GRU layer
        hidden = self.create_hidden(batch_size=inputs.size(1))
        # pack padded batch of sequences for RNN module
        if input_len:
            embedding = pack_padded_sequence(
                input=embedding, lengths=input_len, batch_first=False
            )
        # forward pass through GRU
        outputs, hidden = self.gru(embedding, hidden)
        # unpack padding
        if input_len:
            outputs, _ = pad_packed_sequence(sequence=outputs, batch_first=False)
        # get bi-directional GRU outputs
        hidden = hidden[0] + hidden[1]
        outputs = outputs[:, :, : self.d_model] + outputs[:, :, self.d_model :]

        return outputs.transpose(0, 1), hidden.unsqueeze(0)


class StateGenerator(nn.Module):
    """a GRU decoder to predict the value for each (domain, slot) pairs"""

    def __init__(
        self,
        vocab: CustomVocab,
        shared_embedding,
        vocab_size,
        d_model,
        p_dropout,
        slots: list,
        n_gates,
    ):
        super().__init__()

        self.vocab = vocab
        self.shared_embedding = shared_embedding
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.slots = slots
        self.n_gates = n_gates
        self.dropout = nn.Dropout(p_dropout)
        self.gru = nn.GRU(d_model, d_model, dropout=p_dropout)
        self.w_ratio = nn.Linear(3 * d_model, 1)
        self.sigmoid = nn.Sigmoid()
        self.w_gate = nn.Linear(d_model, n_gates)

        # create independent slot vocab
        self.slot_words_to_ids = {}
        for slot in self.slots:
            # add domain name to slot vocab
            if slot.split("-")[0] not in self.slot_words_to_ids.keys():
                self.slot_words_to_ids[slot.split("-")[0]] = len(self.slot_words_to_ids)
            # add slot name to slot vocab
            if slot.split("-")[1] not in self.slot_words_to_ids.keys():
                self.slot_words_to_ids[slot.split("-")[1]] = len(self.slot_words_to_ids)

        # embedding for domain-slot pairs
        self.pairs_embedding = nn.Embedding(
            num_embeddings=len(self.slot_words_to_ids), embedding_dim=self.d_model
        )
        # initialize domain-slot pairs embedding
        self.pairs_embedding.weight.data.normal_(mean=0, std=0.1)

    def forward(
        self,
        batch_size,
        encoded_hidden: torch.Tensor,
        encoded_outputs,
        encoded_lens,
        story,
        max_resp_len,
        trg_batches,
        b_use_teacher_forcing,
        slots,
    ):
        """
        Args:
            encoded_outputs: encodede dialogue history H_t in paper
        """
        # set decoder outputs
        ptr_outputs = torch.zeros(
            size=(len(slots), batch_size, max_resp_len, self.vocab_size)
        )
        gate_outputs = torch.zeros(size=(len(slots), batch_size, self.n_gates))
        if USE_CUDA:
            ptr_outputs = ptr_outputs.to(device)
            gate_outputs = gate_outputs.to(device)

        # get domain-slot pairs embedding
        slot_embedding_dict = {}
        for i, slot in enumerate(slots):
            # create domain embedding from domain-slot pairs vocab dict
            if slot.split("-")[0] in self.slot_words_to_ids.keys():
                domain_words_to_ids = [self.slot_words_to_ids[slot.split("-")[0]]]
                domain_words_to_ids = torch.tensor(domain_words_to_ids)
                if USE_CUDA:
                    domain_words_to_ids = domain_words_to_ids.to(device)
                domain_embedding = self.pairs_embedding(domain_words_to_ids)

            # create slot embedding from domain-slot pairs vocab dict
            if slot.split("-")[1] in self.slot_words_to_ids.keys():
                slot_words_to_ids2 = [self.slot_words_to_ids[slot.split("-")[1]]]
                slot_words_to_ids2 = torch.tensor(slot_words_to_ids2)
                if USE_CUDA:
                    slot_words_to_ids2 = slot_words_to_ids2.to(device)
                slot_embedding = self.pairs_embedding(slot_words_to_ids2)

            # combine two embeddings as one query
            combined_embedding = domain_embedding + slot_embedding
            slot_embedding_dict[slot] = combined_embedding
            # expand to match tensor size
            expanded_slot_embedding = combined_embedding.expand_as(encoded_hidden)
            if i == 0:
                slot_embedding_array = expanded_slot_embedding.clone()
            else:
                slot_embedding_array = torch.cat(
                    tensors=(slot_embedding_array, expanded_slot_embedding), dim=0
                )

        # get pointer-generator outputs, decoding each domain-slot pair one-by-one
        words_point_out = []
        counter = 0
        for slot in slots:
            words = []
            # domain-slot combined embedding for decoder input
            slot_embedding = slot_embedding_dict[slot]
            # get decoder inputs from domain-slot pairs word embedding
            decoder_inputs = self.dropout(slot_embedding)
            decoder_inputs = decoder_inputs.expand(batch_size, self.d_model)
            for i in range(max_resp_len):
                # get decoder outputs for pointer-generation
                decoder_state, hidden = self.gru(
                    decoder_inputs.expand_as(encoded_hidden), encoded_hidden
                )
                # get history attention through encoder-decoder interaction
                context_vectors, logits, probs = self.get_attention(
                    sequences=encoded_outputs,
                    condition=hidden.squeeze(0),
                    lens=encoded_lens,
                )
                if i == 0:
                    gate_outputs[counter] = self.w_gate(context_vectors)
                # get vocab attention through embedding and decoded hidden states
                p_vocab = self.get_vocab_attention(
                    sequences=self.shared_embedding.weight, condition=hidden.squeeze(0)
                )
                # get domain-slot pair generation probability
                vector_for_p_gen = torch.cat(
                    tensors=[decoder_state.squeeze(0), context_vectors, decoder_inputs],
                    dim=-1,
                )
                # determine generation or copying
                p_gen = self.sigmoid(self.w_ratio(vector_for_p_gen))
                p_context_ptr = torch.zeros(p_vocab.size())
                if USE_CUDA:
                    p_context_ptr = p_context_ptr.to(device)
                # add src to position according to dim-axis
                p_context_ptr.scatter_add_(dim=1, index=story, src=probs)
                # get final output distribution with weighted sum of two distributions
                p_final = (
                    p_gen.expand_as(p_context_ptr) * p_vocab
                    + (1 - p_gen).expand_as(p_context_ptr) * p_context_ptr
                )
                # get word prediction
                pred_word = torch.argmax(p_final, dim=1)
                # get word string from word index
                for word_id in pred_word:
                    # tensor.item() returns only values
                    words.append(self.vocab.ids_to_words[word_id.item()])
                # get ptr outputs
                ptr_outputs[counter, :, i, :] = p_final

                # in case of using teacher forcing
                if b_use_teacher_forcing:
                    # next decoder input is ground truth word
                    decoder_inputs = self.shared_embedding(trg_batches[:, counter, i])
                else:
                    decoder_inputs = self.shared_embedding(pred_word)

            counter += 1
            words_point_out.append(words)

            return ptr_outputs, gate_outputs, words_point_out, []

    def get_attention(self, sequences, condition: torch.Tensor, lens):
        """get attention over the sequences using the condition"""
        scores = condition.unsqueeze(1)
        scores = scores.expand_as(sequences)
        scores = scores.mul(sequences)
        scores = scores.sum(2)
        max_len = max(lens)
        for i, len in enumerate(lens):
            if len < max_len:
                scores.data[i, len:] = -np.inf
        score_probs = F.softmax(scores, dim=1)
        context = scores.unsqueeze(2)
        context = context.expand_as(sequences)
        context = context.mul(sequences)
        context = context.sum(1)

        return context, scores, score_probs

    def get_vocab_attention(self, sequences: torch.Tensor, condition: torch.Tensor):
        sequences = sequences.transpose(1, 0)
        scores = condition.matmul(sequences)
        score_probs = F.softmax(scores, dim=1)

        return score_probs


class TRADE(nn.Module):
    def __init__(
        self,
        d_model,
        vocab: CustomVocab,
        path,
        task,
        lr,
        p_dropout,
        slots,
        slot_gate: dict,
        n_train_vocab=0,
    ):
        super().__init__()

        self.name = "TRADE"

        self.d_model = d_model
        self.vocab = vocab[0]
        self.mem_vocab = vocab[1]
        self.path = path
        self.task = task
        self.lr = lr
        self.p_dropout = p_dropout
        self.slots = slots[0]
        self.slot_temp = slots[2]
        self.slot_gate = slot_gate
        self.n_gates = len(slot_gate)

        self.cross_entropy = nn.CrossEntropyLoss()

        self.encoder = UtteranceEncoder(
            vocab_size=self.vocab.n_words,
            d_model=self.d_model,
            p_dropout=self.p_dropout,
            n_layers=1,
        )
        self.decoder = StateGenerator(
            vocab=self.vocab,
            shared_embedding=self.encoder.embedding,
            vocab_size=self.vocab.n_words,
            d_model=self.d_model,
            p_dropout=self.p_dropout,
            slots=self.slots,
            n_gates=self.n_gates,
        )

        # if pre-trained models exist
        if self.path:
            if USE_CUDA:
                trained_encoder = torch.load(path + "/enc.th")
                trained_decoder = torch.load(path + "/dec.th")
            else:
                trained_encoder = torch.load(
                    path + "/enc.th", map_location=lambda storage, loc: storage
                )
                trained_decoder = torch.load(
                    path + "/dec.th", map_location=lambda storage, loc: storage
                )

            # load pre-trained state dict
            self.encoder.load_state_dict(trained_encoder.state_dict())
            self.decoder.load_state_dict(trained_decoder.state_dict())

        # initialize optimizer
        self.optimizer = optim.Adam(params=self.parameters(), lr=self.lr)
        # initialize lr scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer,
            mode="max",
            factor=0.5,
            patience=1,
            min_lr=0.0001,
            verbose=True,
        )

        # initialize status
        self.reset_status()

        # set GPU environment
        if USE_CUDA:
            self.encoder.to(device)
            self.decoder.to(device)

    def reset_status(self):
        """initialize status"""
        self.loss = 0
        self.print_every = 1
        self.ptr_loss = 0
        self.gate_loss = 0
        self.class_loss = 0

    def print_losses(self):
        """print avg, ptr, and gate losses"""
        print_avg_loss = self.loss / self.print_every
        print_ptr_loss = self.ptr_loss / self.print_every
        print_gate_loss = self.gate_loss / self.print_every
        self.print_every += 1

        return f"AVG LOSS: {print_avg_loss:.2f}, PTR LOSS: {print_ptr_loss:.2f}, GATE LOSS: {print_gate_loss:.2f}"

    def save_model(self, acc: str):
        # set directory to save trained model
        dir = (
            "save/trade_dst"
            + args["add_name"]
            + args["dataset"]
            + str(self.task)
            + "/"
            + "d_model"
            + str(self.hidden_size)
            + "bsz"
            + str(args["batch_size"])
            + "p_dr"
            + str(self.p_dropout)
            + str(acc)
        )
        # check dir path
        if not os.path.exists(dir):
            os.makedirs(dir)
        # save encoder and decoder
        torch.save(self.encoder, dir + "/enc.th")
        torch.save(self.decoder, dir + "/dec.th")

    def train_batch(self, data, slots, reset=True):
        """encode and decode, and get losses"""
        if reset:
            # intialize status
            self.reset_status()

        # initialize optimizer
        self.optimizer.zero_grad()

        # whether to use teacher forcing
        b_use_teacher_forcing = random.random()
        if b_use_teacher_forcing < args.teacher_forcing_ratio:
            b_use_teacher_forcing = True
        else:
            b_use_teacher_forcing = False

        (
            point_outputs,
            gates,
            words_point_out,
            words_class_out,
        ) = self.encode_and_decode(data, b_use_teacher_forcing, slots)

        ptr_loss = masked_cross_entropy_for_value(
            logits=point_outputs.transpose(0, 1).contiguous(),
            # [:, : len(self.point_slots)].contiguous()
            target=data["gen_slotvalue"].contiguous(),
            mask=data["slotvalue_len"],
        )  # [:, : len(self.point_slots)])
        gate_loss = self.cross_entropy(
            input=gates.transpose(0, 1).contiguous().view(-1, gates.size(-1)),
            target=data["gate_label"].contiguous().view(-1),
        )

        # get loss
        if args.b_use_gate:
            loss = ptr_loss + gate_loss
        else:
            loss = ptr_loss

        self.grad_loss = loss
        self.ptr_to_bp_loss = ptr_loss

        # update parameters with optimizers
        self.loss += loss.data
        self.ptr_loss += ptr_loss.item()
        self.gate_loss += gate_loss.item()

    def update_optimizer(self):
        self.grad_loss.backward()
        self.optimizer.step()

    def encode_and_decode(self, data, b_use_teacher_forcing, slots):
        """build unknown mask for memory to encourage generalization"""
        if args.b_unk_mask and self.decoder.training:
            story_size = data["context"].size()
            # create a new array filled with ones
            rand_mask = np.ones(story_size)
            # random sample from bi-nomial distribution
            bi_mask = np.random.binomial(
                [np.ones(shape=(story_size[0], story_size[1]), dtype=np.float)],
                1 - self.p_dropout,
            )[0]
            # get array with random bi-nomial distribution
            rand_mask = rand_mask * bi_mask
            rand_mask = torch.Tensor(rand_mask)
            if USE_CUDA:
                rand_mask = rand_mask.to(device)
            # get data["context"] applied mask
            story = data["context"] * rand_mask.long()
        else:
            story = data["context"]

        # encode dialogue history
        encoded_outputs, encoded_hidden = self.encoder(
            inputs=story.transpose(0, 1), input_len=data["context_len"]
        )

        # get the words that can be copied from the memory (copy mechanism)
        batch_size = len(data["context_len"])
        # copy context from data
        self.copys = data["context_plain"]
        # set max response length
        if self.encoder.training:
            max_resp_len = data["gen_slotvalue"].size(2)
        else:
            max_resp_len = 10

        # pass decoder
        ptr_outputs, gate_outputs, words_point_out, words_class_out = self.decoder(
            batch_size=batch_size,
            encoded_hidden=encoded_hidden,
            encoded_outputs=encoded_outputs,
            encoded_lens=data["context_len"],
            story=story,
            max_resp_len=max_resp_len,
            trg_batches=data["gen_slotvalue"],
            b_use_teacher_forcing=b_use_teacher_forcing,
            slots=slots,
        )

        return ptr_outputs, gate_outputs, words_point_out, words_class_out

    def evaluate_model(self, valid_dl: DataLoader, best_metric: float, slots: list):
        # set to non-training mode to disable dropout
        self.encoder.train(False)
        self.decoder.train(False)

        print("START EVALUATION")

        preds = {}
        inverse_unpoint_slot = {}
        # {"ptr": 0, "dontcare": 1, "none": 2} -> {0: 'ptr', 1: 'dontcare', 2: 'none'}
        for k, v in self.slot_gate.items():
            inverse_unpoint_slot[v] = k
        # get validation dataloader
        progress_bar = tqdm(enumerate(valid_dl), total=len(valid_dl))
        for _, valid_data in progress_bar:
            # len of valid data (list) equals batch size
            batch_size = len(valid_data["context_len"])
            # get outputs from encoder-decoder pair
            _, gate_outputs, word_point_out, _ = self.encode_and_decode(
                data=valid_data, b_use_teacher_forcing=False, slots=slots
            )
            # for every batch
            for batch_idx in range(batch_size):
                # valid_data: {"key_0": list_0, "key_1": list_1, ...}
                if valid_data["ID"][batch_idx] not in preds.keys():
                    preds[valid_data["ID"][batch_idx]] = {}
                # preds: {"ID": {"turn_id": {"turn_belief": [turn_belief value], ...}, ...}, ...}
                preds[valid_data["ID"][batch_idx]][valid_data["turn_id"][batch_idx]] = {
                    "turn_belief": valid_data["turn_belief"][batch_idx]
                }
                pred_belief_batch_size_ptr = []
                gate = torch.argmax(gate_outputs.transpose(0, 1)[batch_idx], dim=1)

                # get pointer-generator results
                # in case of using slot gate
                if args.b_use_gate:
                    for si, sg in enumerate(gate):
                        if sg == self.slot_gate["none"]:
                            continue
                        elif sg == self.slot_gate["ptr"]:
                            # get pred from pointer-generator outputs
                            pred = np.transpose(word_point_out[si])[batch_idx]
                            st = []
                            for e in pred:
                                # in case that predicted word is 'EOS'
                                if e == "EOS":
                                    break
                                else:
                                    st.append(e)
                            # convert list to str with whitespaces
                            st = " ".join(st)
                            # in case that gate equals 'ptr', but predicted word equals 'none'
                            if st == "none":
                                continue
                            # in case that predicted word equals slot name
                            else:
                                # append 'domain-slot' str
                                pred_belief_batch_size_ptr.append(
                                    slots[si] + "-" + str(st)
                                )
                        # in case of sg equals 'dontcare'
                        else:
                            # append 'domain-dontcare' str
                            pred_belief_batch_size_ptr.append(
                                slots[si] + "-" + inverse_unpoint_slot[sg.item()]
                            )
                # in case of not using slot gate
                else:
                    for si, _ in enumerate(gate):
                        # get pred from pointer-generator outputs
                        pred = np.transpose(word_point_out[si])[batch_idx]
                        st = []
                        for e in pred:
                            # in case that predicted word is 'EOS'
                            if e == "EOS":
                                break
                            else:
                                st.append(e)
                        # convert list to str with whitespaces
                        st = " ".join(st)
                        # in case that gate equals 'ptr', but predicted word equals 'none'
                        if st == "none":
                            continue
                        else:
                            # append 'domain-slot' str
                            pred_belief_batch_size_ptr.append(slots[si] + "-" + str(st))
                # add 'pred_bs_ptr' key and value at the last
                preds[valid_data["ID"][batch_idx]][valid_data["turn_id"][batch_idx]][
                    "pred_bs_ptr"
                ] = pred_belief_batch_size_ptr

                # in case that preds do not equal to ground truth
                if (
                    set(valid_data["turn_belief"][batch_idx])
                    != set(pred_belief_batch_size_ptr)
                    and args.b_generate_sample
                ):
                    # notify both preds and ground truth
                    print("True", set(valid_data["turn_belief"][batch_idx]))
                    print("Pred", set(pred_belief_batch_size_ptr), "\n")

        args.b_generate_sample = 1
        if args.b_generate_sample:
            # save predicted results
            with open(file=f"preds_from_{self.name}", mode="w") as f:
                json.dump(preds, f, indent=4)

        # get evaluation metrics
        joint_acc_score_ptr, f1_score_ptr, turn_acc_score_ptr = self.evaluate_metrics(
            preds=preds, from_which_key="pred_bs_ptr", slots=slots
        )

        evaluation_metrics = {
            "Joint Acc": joint_acc_score_ptr,
            "Turn Acc": turn_acc_score_ptr,
            "Joint F1": f1_score_ptr,
        }
        print(evaluation_metrics)

        # Set back to training mode
        self.encoder.train(True)
        self.decoder.train(True)

        joint_acc_score = joint_acc_score_ptr

        if joint_acc_score >= best_metric:
            self.save_model(acc=f"acc_{joint_acc_score:.4f}")
            print("MODEL SAVED")

        return joint_acc_score

    def evaluate_metrics(self, preds: dict, from_which_key: str, slots):
        """calculate metrics"""
        # preset metrics
        total = 0
        turn_acc = 0
        joint_acc = 0
        f1_pred = 0
        f1_count = 0

        for _, values in preds.items():
            for t in range(len(values)):
                cv = values[t]
                # if preds equal to ground truth turn belief state
                if set(cv["turn_belief"]) == set(cv[from_which_key]):
                    joint_acc += 1
                total += 1

                # get slot accuracy
                temp_acc = self.get_accuracy(
                    set(cv["turn_belief"]), set(cv[from_which_key]), slots
                )
                turn_acc += temp_acc

                # Compute prediction joint F1 score
                temp_f1, _, _, count = self.get_confusion_score(
                    set(cv["turn_belief"]), set(cv[from_which_key])
                )
                f1_pred += temp_f1
                f1_count += count

        joint_acc_score = joint_acc / float(total) if total != 0 else 0
        turn_acc_score = turn_acc / float(total) if total != 0 else 0
        f1_score = f1_pred / float(f1_count) if f1_count != 0 else 0

        return joint_acc_score, f1_score, turn_acc_score

    def get_accuracy(self, golds: set, preds: set, slots):
        """"""
        # preset
        n_miss_golds = 0
        miss_slots = []

        for gold in golds:
            if gold not in preds:
                n_miss_golds += 1
                # append missed slot name
                miss_slots.append(gold.rsplit("-", 1)[0])
        # preset
        n_wrong_preds = 0
        for pred in preds:
            if pred not in golds and pred.rsplit("-", 1)[0] not in miss_slots:
                n_wrong_preds += 1
        n_total = len(slots)
        n_corrects = len(slots) - n_miss_golds - n_wrong_preds
        acc = n_corrects / float(n_total)

        return acc

    def get_confusion_score(self, golds, preds):
        """get """
        # preset
        true_pos = 0
        false_pos = 0
        false_neg = 0

        if len(golds) != 0:
            count = 1
            for gold in golds:
                if gold in preds:
                    true_pos += 1
                else:
                    false_neg += 1
            for pred in preds:
                if pred not in golds:
                    false_pos += 1
            # get precision
            precision = (
                true_pos / float(true_pos + false_pos)
                if (true_pos + false_pos) != 0
                else 0
            )
            # get recall
            recall = (
                true_pos / float(true_pos + false_neg)
                if (true_pos + false_neg) != 0
                else 0
            )
            # get F1 score
            f1_score = (
                2 * precision * recall / float(precision + recall)
                if (precision + recall) != 0
                else 0
            )
        else:
            if len(preds) == 0:
                precision = 1
                recall = 1
                f1_score = 1
                count = 1
            else:
                precision = 0
                recall = 0
                f1_score = 0
                count = 1

        return f1_score, recall, precision, count
