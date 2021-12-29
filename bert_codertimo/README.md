# BERT - codertimo ver.
### Summary
* Building BERT model architecture
* Pre-training BERT model
~~~
Code
    ./data/vocab.py : Vocabulary builder for tokenizer
    ./data/dataset.py : Dataset bulider for training, validation, and test datasets
    ./models/attention.py : Transformer layer with attention mechanism
    ./models/bert.py : BERT model architecture
    ./models/embedding.py : Embedding layer for input sequences
    ./models/encoder.py : Encoder layer with one Transformer block
    ./models/language_model.py : Masked language model and next sentence prediction of pre-training
    ./models/lr_scheduler.py : Learning rate scheduler for effective pre-training
    ./models/sub_layer.py : Layer normalization, residual connection, and feed forward layer
    trainer.py : Trainer of model pre-training
    main.py : Executor of pre-training

Database
    kowiki_20211107.txt : Korean wikipedia data for model training (not included)
~~~
This code is based on <https://github.com/codertimo/BERT-pytorch>