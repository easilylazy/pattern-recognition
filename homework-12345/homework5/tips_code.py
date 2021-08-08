import torch
from torchtext.legacy import data
from torchtext.legacy import datasets
from torchtext.vocab import Vectors, GloVe, CharNGram#, FastTex

################################
# DataLoader
################################

# set up fields
TEXT = data.Field()
LABEL = data.Field(sequential=False,dtype=torch.long)

# make splits for data
# DO NOT MODIFY: fine_grained=True, train_subtrees=False
train, val, test = datasets.SST.splits(
    TEXT, LABEL, fine_grained=True, train_subtrees=False)

# print information about the data
print('train.fields', train.fields)
print('len(train)', len(train))
print('vars(train[0])', vars(train[0]))

# build the vocabulary
# you can use other pretrained vectors, refer to https://github.com/pytorch/text/blob/master/torchtext/vocab.py
TEXT.build_vocab(train, vectors=Vectors(name='vector.txt', cache='./data'))
LABEL.build_vocab(train)
# We can also see the vocabulary directly using either the stoi (string to int) or itos (int to string) method.
print(TEXT.vocab.itos[:10])
print(LABEL.vocab.stoi)
print(TEXT.vocab.freqs.most_common(20))

# print vocab information
print('len(TEXT.vocab)', len(TEXT.vocab))
print('TEXT.vocab.vectors.size()', TEXT.vocab.vectors.size())

# make iterator for splits
train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (train, val, test), batch_size=2000)

# print batch information
# batch = next(iter(train_iter)) # for batch in train_iter
# print(batch.text) # input sequence
# print(batch.label) # groud truth

# Attention: batch.label in the range [1,5] not [0,4] !!!

# save format
import numpy as np
import pandas as pd
import csv
with torch.no_grad():
    for i, batch in enumerate(train_iter):
        x=batch.text.transpose(0,1).numpy()#.to(torch.float32)
        y=(batch.label.numpy()-1).reshape(x.shape[0],1)
        d = np.append(y, x, axis=1)
        df = pd.DataFrame(d)#创建随机值
        df.to_csv('train.csv',mode='a', index=False,header=None)
    for i, batch in enumerate(test_iter):
        x=batch.text.transpose(0,1).numpy()#.to(torch.float32)
        y=(batch.label.numpy()-1).reshape(x.shape[0],1)
        d = np.append(y, x, axis=1)
        df = pd.DataFrame(d)#创建随机值
        df.to_csv('test.csv',mode='a', index=False,header=None)
    for i, batch in enumerate(val_iter):
        x=batch.text.transpose(0,1).numpy()#.to(torch.float32)
        y=(batch.label.numpy()-1).reshape(x.shape[0],1)
        d = np.append(y, x, axis=1)
        df = pd.DataFrame(d)#创建随机值
        df.to_csv('val.csv',mode='a', index=False,header=None)


  


################################
# After build your network 
################################


# Copy the pre-trained word embeddings we loaded earlier into the embedding layer of our model.
pretrained_embeddings = TEXT.vocab.vectors

print(pretrained_embeddings.shape)

# you should maintain a nn.embedding layer in your network
# model.embedding.weight.data.copy_(pretrained_embeddings)




