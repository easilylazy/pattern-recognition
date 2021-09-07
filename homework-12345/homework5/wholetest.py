
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

torch.manual_seed(2)
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

pretrained_embeddings = TEXT.vocab.vectors

print(pretrained_embeddings.shape)


# %%

# make iterator for splits
train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (train, val, test), batch_size=64)

# print batch information
batch = next(iter(train_iter)) # for batch in train_iter

# 超参数
max_len = 32 #句子最大长度
embedding_size = 300
hidden_size = 32
batch_size = 2210
label_num = 6
num_layers=2


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
        self.lstm = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size,num_layers=num_layers,dropout=0.8)#,bidirectional=True)
        self.init_w = Variable(torch.Tensor(1, self.hidden_size), requires_grad=True)
        torch.nn.init.uniform_(self.init_w)
        self.init_w = nn.Parameter(self.init_w).to(device)
        self.linear = nn.Linear(self.hidden_size, self.label_num)
        self.criterion  = nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(self.parameters(),lr=1e-3)
    
    def forward(self, input, batch_size):
        input = self.embedding_table(input.long()) # input:[batch_size, max_len, embedding_size]
        h0 = Variable(torch.zeros(num_layers, batch_size, self.hidden_size)).to(device)
        c0 = Variable(torch.zeros(num_layers, batch_size, self.hidden_size)).to(device)
        lstm_out, _ = self.lstm(input.permute(1,0,2),(h0,c0))
        lstm_out = torch.tanh(lstm_out) # [max_len, bach_size, hidden_size]
        M = torch.matmul(self.init_w, lstm_out.permute(1,2,0))
        alpha = F.softmax(M,dim=0)  # [batch_size, 1, max_len]
        out = torch.matmul(alpha, lstm_out.permute(1,0,2)).squeeze() # out:[batch_size, hidden_size]
        predict = F.softmax(self.linear(out)) # out:[batch_size, label_num]
        return predict




# %%
# train_, test_, vocab = processData()
# embedding_table = word_embedding(len(vocab), embedding_size)
train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (train, val, test), batch_size=20)


def feature_scalling(X):
    mmin = X.min()
    mmax = X.max()
    return (X - mmin) / (mmax - mmin), mmin, mmax

x_train, mmin, mmax = feature_scalling(next(iter(train_iter)).text)
print(mmin,mmax)
print(x_train)
# %%

# # print batch information
# net = Classify(len(TEXT.vocab),pretrained_embeddings)
# net=net.to(device)
# net.embedding_table.weight.data.copy_(pretrained_embeddings)
# # embedding_table)
# print(net.embedding_table)
# optim = net.optim

path='pth\\dr__lr_0.01_de_0.01_len_32_hid_32_emb_300_epo_40_bat_15_eval_100_loss_1.401_acc_0.6213.pth'
    # dr__lr_0.01_de_0.01_len_32_hid_32_emb_300_epo_40_bat_15_eval_100_loss_1.5905_acc_0.5297_loss_1.4428_acc_0.5941_loss_1.3352_acc_0.6286_loss_1.1809_acc_0.6295_loss_1.1129_acc_0.649.pth'
net=torch.load(path,map_location=device)
net.eval()
with torch.no_grad():
                print('testing (epoch:',1,')')
                num = 0
                for k in range(1):
                    batch = next(iter(train_iter)) # for batch in train_iter
                    x=batch.text.transpose(0,1).to(torch.float32)
                    print(x)

                    x=Variable(x).to(device)
                    y=batch.label-1
                    x=x.to(device)
                    y=y.to(device)
                    y_hat = net.forward(x, len(x))
                    y_hat = np.argmax(y_hat.cpu().numpy(),axis=1)
                    print(len(np.where((0-y.cpu().numpy())==0)[0]))
                    print(len(np.where((1-y.cpu().numpy())==0)[0]))
                    print(len(np.where((2-y.cpu().numpy())==0)[0]))
                    print(len(np.where((3-y.cpu().numpy())==0)[0]))
                    print(len(np.where((4-y.cpu().numpy())==0)[0]))

                    print(len(np.where((0-y_hat)==0)[0]))
                    print(len(np.where((1-y_hat)==0)[0]))
                    print(len(np.where((2-y_hat)==0)[0]))
                    print(len(np.where((3-y_hat)==0)[0]))
                    print(len(np.where((4-y_hat)==0)[0]))
                    num=len(np.where((y_hat-y.cpu().numpy())==0)[0])
                    print( num,batch_size)
                    acc = round(num/batch_size, 4)
                    print(y)
                    print(y_hat)
                    # if acc > max_acc:
                    #     max_acc = acc
                    print('epoch:', 1, ' | accuracy = ', acc)

batch_size = 2000


train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (train, val, test), batch_size=batch_size)
test_batch=(len(test)//batch_size)
train_batch=(len(train)//batch_size)
total_test=batch_size*test_batch
total_train=batch_size*train_batch
loss_list=[]
acc_list=[]
ej = 0

batch = next(iter(test_iter)) # for batch in train_iter
print(len(test_iter))
print(len(train_iter))
# for i in range(len(test_iter)):
#     print(test_iter[i][:9])
with torch.no_grad():
                print('testing (epoch:',10,')')
                num = 0
                record=np.zeros(5)
                for i, batch in enumerate(train_iter):
                    x=batch.text.transpose(0,1).to(torch.float32)
                    x=Variable(x).to(device)
                    y=batch.label-1
                    print(y[:19])
                    x=x.to(device)
                    y=y.to(device)
                    y_hat = net.forward(x, len(x))
                    y_hat = np.argmax(y_hat.cpu().numpy(),axis=1)
                    num=len(np.where((y_hat-y.cpu().numpy())==0)[0])
                    print(num)
                    record[0]+=(len(np.where((0-y.cpu().numpy())==0)[0]))
                    record[1]+=(len(np.where((1-y.cpu().numpy())==0)[0]))
                    record[2]+=(len(np.where((2-y.cpu().numpy())==0)[0]))
                    record[3]+=(len(np.where((3-y.cpu().numpy())==0)[0]))
                    record[4]+=(len(np.where((4-y.cpu().numpy())==0)[0]))
                print( num,total_test)
                print(record)
                acc = round(num/total_test, 4)
                print('epoch:', 10, ' | accuracy = ', acc)