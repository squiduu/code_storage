from bert import BertModel
from config import config
from pretrained_transplantor import set_learned_params
import torch.nn as nn
import torch
import torch.optim as optim
import time
from dataloader import dl_dict, test_dl
from tqdm import tqdm

# sets model architecture
net_bert = BertModel(config)
# transplants pre-trained parameters
net_bert = set_learned_params(net_bert, weights_path="./weights/pytorch_model.bin")


class BertForIMDb(nn.Module):
    """model that connects the parts that judge whether the data is positive or negative"""

    def __init__(self, net_bert):
        super().__init__()

        # sets model
        self.bert = net_bert
        # sets FCL for classification
        self.cls = nn.Linear(config.hidden_size, 2)
        # initializes FCL
        nn.init.normal_(self.cls.weight, std=0.02)
        nn.init.uniform_(self.cls.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids=None,
        attention_mask=None,
        output_all_encoded_layers=False,
        attention_show_flg=False,
    ):
        """
        Args:
            input_ids: tokenized input IDs for subwords, [batch_size, seq_len]
        """

        # forward propagation
        if attention_show_flg is True:
            encoded_layers, _, attention_probs = self.bert(
                input_ids,
                token_type_ids,
                attention_mask,
                output_all_encoded_layers,
                attention_show_flg,
            )
        else:
            encoded_layers, _ = self.bert(
                input_ids,
                token_type_ids,
                attention_mask,
                output_all_encoded_layers,
                attention_show_flg,
            )

        # classifies using representation vectors of [CLS]
        vec_0 = encoded_layers[:, 0, :]
        # converts size to [batch_size, hidden_size]
        vec_0 = vec_0.view(-1, config.hidden_size)
        # applies classfication FCL
        out = self.cls(vec_0)

        # gets attention probabilities
        if attention_show_flg is True:
            return out, attention_probs
        else:
            return out


# builds model
net = BertForIMDb(net_bert)
# sets training mode
net.train()
print("NETWORK SETTING COMPLETED")

# finetunes only the last transformer layer
# sets all backpropagations as false
for _, param in net.named_parameters():
    param.requires_grad = False
# sets backpropagation of the last transformer layer as true
for _, param in net.bert.encoder.layer[-1].named_parameters():
    param.requires_grad = True
# sets backpropagation of the classification FCL as true
for _, param in net.cls.named_parameters():
    param.requires_grad = True

# sets optimization
optimizer = optim.Adam(
    [
        {"params": net.bert.encoder.layer[-1].parameters(), "lr": 5e-5},
        {"params": net.cls.parameters(), "lr": 5e-5},
    ],
    betas=(0.9, 0.999),
)

# sets loss function
criterion = nn.CrossEntropyLoss()


def train_model(net: nn.Module, dl_dict, n_epochs, optimizer: optim.Adam, criterion):
    """finetunes parameters of model"""

    # sets GPU environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # notifies the current status
    print("DEVICE:", device)
    print("----------START----------")

    # sends network to GPU environment
    net.to(device)

    # activates using optimized algorithm
    torch.backends.cudnn.benchmark = True
    # deactivates ensuring reproducibility
    torch.backends.cudnn.deterministic = False

    # sets mini-batch size
    batch_size = dl_dict["train"].batch_size

    # for each epoch
    for epoch in range(n_epochs):
        # sets training or validation mode
        for mode in ["train", "val"]:
            if mode == "train":
                net.train()
            else:
                net.eval()

            # sets metrics
            epoch_loss = 0.0
            epoch_corrects = 0
            # sets iteration number
            iteration = 1

            # check starting time of training
            time_iteration_start = time.time()

            # loads mini-batch data from dataloader
            for batch in dl_dict[mode]:
                # sends data to GPU environment
                inputs = batch.Text[0].to(device)
                labels = batch.Label.to(device)

                # initializes optimizer
                optimizer.zero_grad()

                # runs forward propagation
                with torch.set_grad_enabled(mode == "train"):
                    # gets output of network
                    outputs = net(
                        inputs,
                        token_type_ids=None,
                        attention_mask=None,
                        output_all_encoded_layers=False,
                        attention_show_flg=False,
                    )
                    # calculates loss
                    loss = criterion(outputs, labels)

                    # predict the labels
                    _, preds = torch.max(outputs, 1)

                    # runs backpropagation
                    if mode == "train":
                        # gets gradients
                        loss.backward()
                        # updates parameters
                        optimizer.step()

                        # for each 10-iteration
                        if iteration % 10 == 0:
                            # gets ending time of training
                            time_iteration_end = time.time()
                            duration = time_iteration_end - time_iteration_start

                            # calculates accuracy
                            acc = torch.sum(preds == labels.data).double() / batch_size

                            # check starting time of training
                            time_iteration_start = time.time()

                            # notifies training status per 10-iterations
                            print(
                                f"ITERATION: {iteration} || LOSS: {loss.item():.4f} || DURATION: {duration:.2f}sec || ACC: {acc:.4f}"
                            )

                    # updates iteration number
                    iteration += 1

                    # calculates total loss and total corrects during an epoch
                    epoch_loss += loss.item() * batch_size
                    epoch_corrects += torch.sum(preds == labels.data)

            # calculates mean loss and accuracy per epoch
            epoch_loss = epoch_loss / len(dl_dict[mode].dataset)
            epoch_acc = epoch_corrects.double() / len(dl_dict[mode].dataset)

            # notifies traininig status per epoch
            print(
                f"EPOCH: {epoch + 1}/{n_epochs} | {mode:^5s} | LOSS: {epoch_loss:.4f} | ACC: {epoch_acc:.4f}"
            )

        # saves fine-tuned model
        MODEL_PATH = "./checkpoints/batchsize_64/"
        torch.save(net, MODEL_PATH + f"finetuned_ep{epoch + 1}_acc{epoch_acc:.4f}.pt")

        WEIGHTS_PATH = "./weights/finetuning/batchsize_64/"
        torch.save(
            net.state_dict(),
            WEIGHTS_PATH + f"finetuned_ep{epoch + 1}_acc{epoch_acc:.4f}.pt",
        )
        print(f"NETWORK & WEIGHTS SAVED: EPOCH {epoch + 1}/{n_epochs}")

    return net


############
# TRAINING #
############
n_epochs = 2
net_trained = train_model(
    net=net,
    dl_dict=dl_dict,
    n_epochs=n_epochs,
    optimizer=optimizer,
    criterion=criterion,
)

########
# TEST #
########
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# sets fine-tuned model as evaluation mode
net_trained.eval()
# sends fine-tuned model to GPU environment
net_trained.to(device)

# the number of correct predictions
epoch_corrects = 0
for batch in tqdm(test_dl):
    inputs = batch.Text[0].to(device)
    labels = batch.Label.to(device)

    # only forward propagation
    with torch.set_grad_enabled(False):
        # gets output of model
        outputs = net_trained(
            input_ids=inputs,
            token_type_ids=None,
            attention_mask=None,
            output_all_encoded_layers=False,
            attention_show_flg=False,
        )

        # calculate loss
        loss = criterion(outputs, labels)
        # get predict results
        _, preds = torch.max(outputs, 1)
        # calculate the number of correct predictions
        epoch_corrects += torch.sum(preds == labels.data)

# calculate accuracy
epoch_acc = epoch_corrects.double() / len(test_dl.dataset)

print(f"TEST ACC: {epoch_acc:.4f}")
