#!/usr/bin/env python
# coding: utf-8

from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe


RAW = data.RawField()
TEXT = data.Field(batch_first=True)
LABEL = data.Field(sequential=False, unk_token=None)

train, dev, test = data.TabularDataset.splits(
    path='.data/quora',
    train='train.tsv',
    validation='dev.tsv',
    test='test.tsv',
    format='tsv',
    fields=[('label', LABEL),
            ('q1', TEXT),
            ('q2', TEXT),
            ('id', RAW)])

TEXT.build_vocab(train, dev, test, vectors=GloVe(name='840B', dim=300))
LABEL.build_vocab(train)

sort_key = lambda x: data.interleave_keys(len(x.q1), len(x.q2))

train_iter, dev_iter, test_iter = \
    data.BucketIterator.splits((train, dev, test),
                               batch_sizes=[32] * 3,
                               device=True,
                               sort_key=sort_key)

print(vars(TEXT.vocab))
max_word_len = max([len(w) for w in TEXT.vocab.itos])
# for <pad>
char_vocab = {'': 0}
# for <unk> and <pad>
characterized_words = [[0] * max_word_len, [0] * max_word_len]
