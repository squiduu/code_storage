import torch
import argparse

# set constant variables
UNK_TOKEN = 0
PAD_TOKEN = 1
EOS_TOKEN = 2
SOS_TOKEN = 3
MAX_LEN = 10

# whether to use cuda or not
if torch.cuda.is_available():
    USE_CUDA = True
    device = "cuda:1"
else:
    USE_CUDA = False
    device = "cpu"
print(device)

parser = argparse.ArgumentParser(description="TRADE-DST")
# training setting
parser.add_argument("--dataset", required=False, default="multiwoz")
parser.add_argument("--task", required=False, default="dst")
parser.add_argument("--path", required=False, default=None)
parser.add_argument("--n_samples", required=False, default=None)
parser.add_argument("--patience", required=False, default=6, type=int)
parser.add_argument("--early_stop", required=False, default="BLEU")
parser.add_argument("--b_all_vocab", required=False, default=1, type=int)
parser.add_argument("--b_imbalance_sampler", required=False, default=0, type=int)
parser.add_argument("--data_ratio", required=False, default=100, type=int)
parser.add_argument("--b_unk_mask", required=False, default=1, type=int)
parser.add_argument("--batch_size", required=False, type=int)
parser.add_argument("--n_epochs", required=False, default=200, type=int)
# testing setting
parser.add_argument("--b_valid_test", required=False, default=0, type=int)
parser.add_argument("--b_visualization", required=False, default=0, type=int)
parser.add_argument("--b_generate_sample", required=False, default=0, type=int)
parser.add_argument("--eval_period", required=False, default=1)
parser.add_argument("--add_name", required=False, default="")
parser.add_argument("--eval_batch_size", required=False, default=0, type=int)
# model architecture setting
parser.add_argument("--b_use_gate", required=False, default=1, type=int)
parser.add_argument("--b_load_embedding", required=False, default=0, type=int)
parser.add_argument("--b_fix_embedding", required=False, default=0, type=int)
parser.add_argument("--b_parallel_decode", required=False, default=0, type=int)
# model hyper-parameters
parser.add_argument("--decoder", required=False)
parser.add_argument("--d_model", required=False, default=400, type=int)
parser.add_argument("--lr", required=False, type=float)
parser.add_argument("--p_dropout", required=False, type=float)
parser.add_argument("--word_limit", required=False, default=-10000)
parser.add_argument("--grad_clipping", required=False, default=10, type=int)
parser.add_argument("--teacher_forcing_ratio", required=False, default=0.5, type=float)

# args = parser.parse_args()
args = parser.parse_args()

# set options
if args.b_load_embedding:
    args.d_model = 400
    print("[Warning] Using d_model=400 for pretrained word embedding (300 + 100)...")
if args.b_fix_embedding:
    args.add_name += "fix_emb"
