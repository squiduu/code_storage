import torch.nn as nn
from iterate_data import get_dataloader_and_text
from transformer import TransformerClassification
import torch.optim as optim
import torch


# get dataloader
train_dl, val_dl, test_dl, TEXT = get_dataloader_and_text(
    max_length=256, batch_size=128
)

# set dataloader dictionary object for dataloader seperation
dl_dict = {"train": train_dl, "val": val_dl}

# set model
net = TransformerClassification(
    embed_vec=TEXT.vocab.vectors, d_model=300, max_seq_len=256, d_output=2
)


# initialize the network
# m equals submodules of the network
def init_weights(m):
    # refer to class name
    classname = m.__class__.__name__
    # for every Linear layer in model
    # str.find() returns -1 if str does not exist
    if classname.find("Linear") != -1:
        # initialize linear layer weights with He init
        nn.init.kaiming_normal_(tensor=m.weight)
        if m.bias is not None:
            # initialize linear layer bias with 0.0
            nn.init.constant_(tensor=m.bias, val=0.0)


# set training mode as default
net.train()

# initialize all the layers
# init_weights function is applied to all the submodules with torch.nn.Module.apply(func)
net.net_3_1.apply(init_weights)
net.net_3_2.apply(init_weights)

# notify that network setup is completed
print("NETWORK SETUP COMPLETED.")

# set loss function
criterion = nn.CrossEntropyLoss()

# set optimizer
lr = 2e-5
optimizer = optim.Adam(params=net.parameters(), lr=lr)


def train_model(
    net: nn.Module,
    dl_dict: dict,
    criterion: nn.CrossEntropyLoss,
    optimizer: optim.Adam,
    n_epochs: int,
):
    """set the model trainer function"""

    # set GPU environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # notify that using GPU or CPU
    print("DEVICE:", device)
    # notify that start of training
    print("----- TRAINING START -----")

    # set the network to GPU environment
    net = net.to(device)

    # accelerate training with embedded cudnn automatic tuner as active
    torch.backends.cudnn.benchmark = True

    # for each epoch
    for epoch in range(n_epochs):
        # for each mode
        for mode in ["train", "val"]:
            if mode == "train":
                net.train()
            else:
                net.eval()
            # initialize sum of epoch loss
            epoch_loss = 0.0
            # initialize the number of correct answers
            epoch_corrects = 0

            # get mini-batch data from dataloader dictionary
            # batch is dictionary object of text and labels
            for batch in dl_dict[mode]:
                # set input text and labels to GPU environment
                # Text and Label are field name of dataloader
                inp_text = batch.Text[0].to(device)
                labels = batch.Label.to(device)

                # initialize optimizer to zero
                optimizer.zero_grad()

                # forward propagation
                # track history if only in training mode
                with torch.set_grad_enabled(mode=(mode == "train")):
                    # special token '<pad>' equals 1 in vocab ID
                    # set attention padding mask with '<pad>'
                    inp_mask = inp_text != 1

                    # get the final classification results
                    y, _, _ = net(inp_text, inp_mask)
                    # get loss
                    loss = criterion(y, labels)
                    # predict labels with row: 0 or 1
                    _, preds = torch.max(y, dim=1)

                    # set backpropagation if only in training mode
                    if mode == "train":
                        # get gradients for all the parameters of networks
                        loss.backward()
                        # update parameters of optimizer with given gradients
                        optimizer.step()

                    # get training loss and corrects for labels
                    epoch_loss += loss.item() * inp_text.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)

            # get epoch loss and accuracy
            epoch_loss = epoch_loss / len(dl_dict[mode].dataset)
            epoch_acc = epoch_corrects.double() / len(dl_dict[mode].dataset)

            # notify training status
            print(
                f"EPOCH {epoch + 1}/{n_epochs} | {mode:^5s} | LOSS: {epoch_loss:.4f}, | ACC: {epoch_acc:.4f}"
            )

        # set path to save network
        PATH = "./checkpoints/batchsize_128/"
        # save model
        torch.save(net, PATH + f"model_ep{epoch + 1}_acc{epoch_acc:.4f}.pt")
        print(f"NETWORK SAVED: EPOCH {epoch + 1}/{n_epochs}")

    return net


# set the number of epochs
n_epochs = 10
# train the model
net_trained = train_model(
    net=net,
    dl_dict=dl_dict,
    criterion=criterion,
    optimizer=optimizer,
    n_epochs=n_epochs,
)
