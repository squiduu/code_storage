# BERT - codertimo ver. - PyTorch
### Summary
* Building BERT model architecture
* Pre-training BERT model
~~~
Code
    ./data/dataset.py : Dataset maker for model training or inference
    ./data/vocab.py : Vocab maker for model training or inference
    ./models/embedding.py : Embedding layer of input sequences
    ./models/attention.py : Attention mechanism for transformer layer
    ./models/sub_layer.py : Layer normalizations, residual connections, feed forward layers in transformer blocks
    ./models/encoder.py : Encoder layer made by stacking transformer layers
    ./models/bert.py : Overall model architecture
    ./models/language_model.py : Masked language model and next sentence prediction for pre-training
    ./models/lr_scheduler.py : Learning rate scheduler for effective pre-training
    trainer.py : Pre-trainer BERT model
    main.py : Main file for pre-training BERT
~~~
This code is based on <https://github.com/codertimo/BERT-pytorch>
