# Multi-Domain and Unseen-Domain Dialogue State Tracking - Chien-Sheng Wu et al.,
### Summary
* Generating dialogue state from utterances using a copy mechanism
* Model is composed of an utterance encoder, a slot gate, and a state generator
~~~
Code
    config.py : Configuration setting for training and evaluation
    create_data.py : Preprocessing raw dataset and making vocabulary for training
    fix_label.py : Fixing annotations or typos
    masked_cross_entropy.py : Calculating loss function for training
    model.py : Architecture of overall model
    multiwoz_dst.py : Dataset, dataloader, and tokenizers for training
    trainer.py : Training and updating parameters

Vocab
    Making vocabulary only with words of MultiWOZ 2.0 dataset

Database
    MultiWOZ 2.0 : The most famous dialogue dataset for common DST models  (not included)
~~~
This code is based on <https://github.com/jasonwu0731/trade-dst>