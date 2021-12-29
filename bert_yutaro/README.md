# BERT - Yutaro Ogawa ver. - PyTorch
### Summary
* Building BERT model architecture
* Fine-tuning of pre-trained BERT
~~~
Code
    config.py : Configuration for model training or inference
    tokenizer.py : Tokenizer of input sequences
    dataloader.py : Dataloader for training, validation, and test datasets
    layer_norm.py : Layer normalization in transformer blocks
    embedding.py : Embedding layer of input sequences
    transformer.py : Transformer layer with attention mechanism
    encoder.py : Encoder layer made by stacking transformer layers
    pooler.py : Pooler layer for classification using the last representation
    bert.py : Overall model architecture
    pretrained_transplantor.py : Transplantor of pre-trained parameters
    finetuner.py : Fine-tuning for IMDb dataset

Vocab
    ./vocab/bert-base-uncased-vocab.txt : Vocabulary set for model training

Config
    ./weights/bert_config.py : a configuration for model
~~~
This code is based on <https://github.com/YutaroOgawa/pytorch_advanced>
