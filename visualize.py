from IPython.display import HTML
from iterate_data import get_dataloader_and_text
import torch


def highlight(word, attn):
    """the higher attention value, the stronger red background of text"""

    html_color = "#%02X%02X%02X" % (255, int(255 * (1 - attn)), int(255 * (1 - attn)))

    return f'<span style="background-color: {html_color}"> {word}</span>'


def make_html(index, batch, preds, normalized_weights_1, normalized_weights_2, TEXT):
    """make html data"""

    # get index result of sentences, labels, and preds
    sentence = batch.Text[0][index]
    label = batch.Label[index]
    pred = preds[index]

    # get and normalize attention values of index
    attns_1 = normalized_weights_1[index, 0, :]
    attns_1 /= attns_1.max()

    attns_2 = normalized_weights_2[index, 0, :]
    attns_2 /= attns_2.max()

    # replace labels and predictions to word
    if label == 0:
        label_str = "Negative"
    else:
        label_str = "Positive"

    if pred == 0:
        pred_str = "Negative"
    else:
        pred_str = "Positive"

    # make html data to show
    html = f"Correct labels: {label_str}<br>Inference labels: {pred_str}<br><br>"

    # get attentions of the first transformer block
    html += "[Visualization attentions of the first transformer block]<br>"
    for word, attn in zip(sentence, attns_1):
        html += highlight(TEXT.vocab.itos[word], attn)
    html += "<br><br>"

    # get attentions of the second transformer block
    html += "[Visualization attentions of the first transformer block]<br>"
    for word, attn in zip(sentence, attns_2):
        html += highlight(TEXT.vocab.itos[word], attn)
    html += "<br><br>"

    return html


# set batch size
batch_size = 128
# get dataloader
train_dl, val_dl, test_dl, TEXT = get_dataloader_and_text(
    max_length=256, batch_size=batch_size
)
# get mini-batch from dataloader
batch = next(iter(test_dl))

# set GPU environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set input text and labels to GPU environment
# Text and Label are field name of dataloader
inp_text = batch.Text[0].to(device)
labels = batch.Label.to(device)

# special token '<pad>' equals 1 in vocab ID
# set attention padding mask with '<pad>'
inp_mask = inp_text != 1

# set path of trained models
PATH = "./checkpoints/"
# load trained model
net_trained = torch.load(f=PATH + f"batchsize_{batch_size}/model_ep10_acc0.8550.pt")
# set evaluation mode
net_trained.eval()
# set trained model to GPU environment
net_trained = net_trained.to(device)

# get the final classification results
y, normalized_weights_1, normalized_weights_2 = net_trained(inp_text, inp_mask)
# predict labels with row: 0 or 1
_, preds = torch.max(y, dim=1)

# get data to show
index = 3
# make html
html_outp = make_html(
    index=index,
    batch=batch,
    preds=preds,
    normalized_weights_1=normalized_weights_1,
    normalized_weights_2=normalized_weights_2,
    TEXT=TEXT,
)
# visualize html result
HTML(html_outp)
