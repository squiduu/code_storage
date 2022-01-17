from config import *
from multiwoz_dst import *
from model import *
from tqdm import tqdm

# model configurations
best_acc = 0.0
counts = 0
accuracy = 0.0

# load dataloader
(
    train_dl,
    valid_dl,
    test_dl,
    vocab,
    total_slots,
    slot_gate,
    n_train_vocab,
) = prepare_dataloader(training=True, task=args.task, batch_size=args.batch_size)

# set model
model = TRADE(
    d_model=args.d_model,
    vocab=vocab,
    path=args.path,
    task=args.task,
    lr=args.lr,
    p_dropout=args.p_dropout,
    slots=total_slots,
    slot_gate=slot_gate,
    n_train_vocab=n_train_vocab,
)

# start training
for epoch in range(args.n_epochs):
    print(f"CURRENT EPOCH: {epoch}")
    # set tqdm for dataloader
    progress_bar = tqdm(enumerate(train_dl), total=len(train_dl))
    for i, data in progress_bar:
        # train per batch and get losses
        model.train_batch(data=data, slots=total_slots[1], reset=False)
        # update optimizer
        model.update_optimizer()
        # set description of the progress bar, desc: str
        progress_bar.set_description(desc=model.print_losses())

    # for every evaludation period
    if (epoch + 1) % int(args.eval_period) == 0:
        acc = model.evaluate_model(
            valid_dl=valid_dl, best_metric=best_acc, slots=total_slots[2]
        )
        model.scheduler.step(acc)

        if acc >= best_acc:
            # update best_acc
            best_acc = acc
            # initialize counts if acc outperforms the previous one
            counts = 0
            best_model = model
        else:
            # increase counts if acc does not improved
            counts += 1

        if counts == args.patience or acc == 1.0:
            break
