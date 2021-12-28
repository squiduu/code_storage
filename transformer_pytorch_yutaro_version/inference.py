import torch
from iterate_data import get_dataloader_and_text


# set GPU environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set batch size
batch_size = 256
# set path of trained models
PATH = "./checkpoints/"
# load trained model
net_trained = torch.load(f=PATH + f"batchsize_{batch_size}/model_ep1_acc0.6428.pt")
# set evaluation mode
net_trained.eval()
# set trained model to GPU environment
net_trained = net_trained.to(device)

# set correct metric
epoch_corrects = 0

# get dataloader
train_dl, val_dl, test_dl, TEXT = get_dataloader_and_text(
    max_length=256, batch_size=batch_size
)

# for each mini-batch data from test dataloader
for batch in test_dl:
    # set input text and labels to GPU environment
    # Text and Label are field name of dataloader
    inp_text = batch.Text[0].to(device)
    labels = batch.Label.to(device)

    # forward propagation
    with torch.set_grad_enabled(False):
        # special token '<pad>' equals 1 in vocab ID
        # set attention padding mask with '<pad>'
        inp_mask = inp_text != 1

        # get the final classification results
        y, _, _ = net_trained(inp_text, inp_mask)
        # predict labels with row: 0 or 1
        _, preds = torch.max(y, dim=1)

        # get correct metric
        epoch_corrects += torch.sum(preds == labels.data)

# get accuracy metric
# Tensor.double() is equivalent to Tensor.to(torch.float64)
epoch_acc = epoch_corrects.double() / len(test_dl.dataset)

# notify evaluation result
print(f"TEST ACC: {epoch_acc:.4f} FOR {len(test_dl.dataset)} TEST DATASET")
