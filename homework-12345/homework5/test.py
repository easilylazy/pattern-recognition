# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'
# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchtext.legacy import data
from torchtext.legacy import datasets
from torchtext.vocab import Vectors, GloVe, CharNGram#, FastTex
from torch.autograd import Variable

import numpy as np
from os import stat
import sys, getopt

torch.manual_seed(1)


# %%
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




# %%

# build the vocabulary
# you can use other pretrained vectors, refer to https://github.com/pytorch/text/blob/master/torchtext/vocab.py
TEXT.build_vocab(train, vectors=Vectors(name='vector.txt', cache='./data'))
LABEL.build_vocab(train)
# We can also see the vocabulary directly using either the stoi (string to int) or itos (int to string) method.




# %%

# make iterator for splits
train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (train, val, test), batch_size=64)

# print batch information
batch = next(iter(train_iter)) # for batch in train_iter

# Attention: batch.label in the range [1,5] not [0,4] !!!


# %%


################################
# After build your network 
################################


# Copy the pre-trained word embeddings we loaded earlier into the embedding layer of our model.
pretrained_embeddings = TEXT.vocab.vectors

print(pretrained_embeddings.shape)


# %%
# 超参数
# max_len = 65 #句子最大长度
# embedding_size = 300
# hidden_size = 100
# batch_size = 64
# epoch = 100
# label_num = 6
# eval_time = 15 # 每训练100个batch后对测试集或验证集进行测试
from param import get_param
info_str,max_len ,embedding_size ,hidden_size ,batch_size,epoch,label_num ,eval_time =get_param()


# %%
class Classify(nn.Module):
    def __init__(self,vocab_len, embedding_table):
        super(Classify, self).__init__()
        self.max_len = max_len
        self.batch_size = batch_size
        # 这里我只是默认初始化词向量，也可以使用torch.from_numpy来加载预训练词向量
        self.embedding_table = nn.Embedding(vocab_len, embedding_size)
        self.embedding_size = embedding_size
        self.hidden_size= hidden_size
        self.label_num = label_num
        self.lstm = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size,num_layers=1,dropout=0.8,bidirectional=True)
        self.init_w = Variable(torch.Tensor(1, 2*self.hidden_size), requires_grad=True)
        self.init_w = nn.Parameter(self.init_w)
        self.linear = nn.Linear(2*self.hidden_size, self.label_num)
        self.criterion  = nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(self.parameters())
    
    def forward(self, input, batch_size):
        input = self.embedding_table(input.long()) # input:[batch_size, max_len, embedding_size]
        h0 = Variable(torch.zeros(2, batch_size, self.hidden_size))
        c0 = Variable(torch.zeros(2, batch_size, self.hidden_size))
        lstm_out, _ = self.lstm(input.permute(1,0,2),(h0,c0))
        lstm_out = F.tanh(lstm_out) # [max_len, bach_size, hidden_size]
        M = torch.matmul(self.init_w, lstm_out.permute(1,2,0))
        alpha = F.softmax(M,dim=0)  # [batch_size, 1, max_len]
        out = torch.matmul(alpha, lstm_out.permute(1,0,2)).squeeze() # out:[batch_size, hidden_size]
        predict = F.softmax(self.linear(out)) # out:[batch_size, label_num]
        return predict




# %%
# train_, test_, vocab = processData()
# embedding_table = word_embedding(len(vocab), embedding_size)
train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (train, val, test), batch_size=batch_size)

# %%

# print batch information
net = Classify(len(TEXT.vocab),pretrained_embeddings)
net.embedding_table.weight.data.copy_(pretrained_embeddings)
# embedding_table)
print(net.embedding_table)
optim = net.optim
max_acc = 0.0 # 记录最大准确率的值
ej = 0

test_batch=total=(len(test)//batch_size)
total_test=batch_size*(len(test)//batch_size)
loss_list=[]
acc_list=[]
for i in range(epoch):
    # batch = next(iter(train_iter)) # for batch in train_iter
    # batch_it = batch(train_, 'train')
    print('training (epoch:',i+1,')')
    for i in range(batch_size):
    # for sentence, tags in batch:
        batch = next(iter(train_iter)) # for batch in train_iter

        x=batch.text.transpose(0,1).to(torch.float32)
        y=batch.label-1
        ej += 1
        y_hat = net.forward(Variable(torch.Tensor(x)), len(x))
        # y = torch.max(torch.Tensor(y), 1)[1]
        loss = net.criterion(y_hat, y)
        if (ej+1)%50 == 0:
            print('epoch:', i+1, ' | batch' , ej ,' | loss = ', loss)
        net.optim.zero_grad()    
        loss.backward(retain_graph=True)
        net.optim.step()
        loss_list.append(loss.item())
        
        if ej%eval_time == 0:    
            # 测试
            with torch.no_grad():
                print('testing (epoch:',i+1,')')
                num = 0
                for i in range(test_batch):
                    batch = next(iter(train_iter)) # for batch in train_iter
                    x=batch.text.transpose(0,1).to(torch.float32)
                    y=batch.label-1
                    y_hat = net.forward(Variable(torch.Tensor(x)), len(x))
                    y_hat = np.argmax(y_hat.numpy(),axis=1)
                    num+=len(np.where((y_hat-y.numpy())==0)[0])
                acc = round(num/total_test, 4)
                acc_list.append(acc)
                if acc > max_acc:
                    max_acc = acc
                print('epoch:', i+1, ' | accuracy = ', acc, ' | max_acc = ', max_acc)




info_str+='_loss_'+str(round(loss.item(),4))+'_acc_'+str(max_acc)
from plot import plot_loss_and_acc

plot_loss_and_acc({'res': [loss_list, acc_list]},show=False,title=info_str)