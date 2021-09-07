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
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
_, _, test_iter = data.BucketIterator.splits(
    (train, val, test), batch_size=2210)
_, val_iter, _ = data.BucketIterator.splits(
    (train, val, test), batch_size=1101)



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
info_str,max_len ,embedding_size ,hidden_size ,batch_size,epoch,label_num ,eval_time,learning_rate,weight_decay =get_param()
info_str='bidr_'+info_str
num_layers=2
drop_out=0
bidirectional= True
if bidirectional:
    total_layers=num_layers*2
else:
    total_layers=num_layers
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
        if bidirectional:
            self.total_hidden_size=2*self.hidden_size
        else:
            self.total_hidden_size=self.hidden_size
        self.label_num = label_num
        self.lstm = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size,num_layers=num_layers,dropout=drop_out,bidirectional=bidirectional)
        self.init_w = Variable(torch.Tensor(1, self.total_hidden_size), requires_grad=True)
        torch.nn.init.normal_ (self.init_w, mean=0, std=1)
        self.init_w = nn.Parameter(self.init_w).to(device)
        self.linear = nn.Linear(self.total_hidden_size, self.label_num)
        self.criterion  = nn.CrossEntropyLoss().to(device)
        self.optim = torch.optim.Adamax(self.parameters(),lr=learning_rate,weight_decay=0)
    
    def forward(self, input, batch_size):
        input = self.embedding_table(input.long()) # input:[batch_size, max_len, embedding_size]
        h0 = Variable(torch.zeros(total_layers, batch_size, self.hidden_size)).to(device)
        c0 = Variable(torch.zeros(total_layers, batch_size, self.hidden_size)).to(device)
        lstm_out, _ = self.lstm(input.permute(1,0,2),(h0,c0))
        # out = torch.tanh(lstm_out) # [max_len, bach_size, hidden_size]
        lstm_out = torch.tanh(lstm_out) # [max_len, bach_size, hidden_size]
        M = torch.matmul(self.init_w, lstm_out.permute(1,2,0))
        alpha = F.softmax(M,dim=0)  # [batch_size, 1, max_len]
        out = torch.matmul(alpha, lstm_out.permute(1,0,2)).squeeze() # out:[batch_size, hidden_size]
        predict = F.softmax(self.linear(out),dim=1) # out:[batch_size, label_num]
        return predict




# %%
# train_, test_, vocab = processData()
# embedding_table = word_embedding(len(vocab), embedding_size)
train_iter, val_iter, _ = data.BucketIterator.splits(
    (train, val, test), batch_size=batch_size,shuffle=True)
test_batch = next(iter(test_iter)) # for batch in train_iter
val_batch = next(iter(val_iter)) # for batch in train_iter

# %%

# print batch information
net = Classify(len(TEXT.vocab),pretrained_embeddings)
net=net.to(device)
net.embedding_table.weight.data.copy_(pretrained_embeddings)
# embedding_table)
print(net.embedding_table)
optim = net.optim
max_acc = 0.0 # 记录最大准确率的值
val_max_acc = 0.0 # 记录最大准确率的值
val_max_num=0

total_test=len(test)
total_train=len(train)
loss_list=[]
acc_list=[]
ej = 0
for i in range(epoch):
    print('training (epoch:',i+1,')')
    train_num=0
    for j, batch in enumerate(train_iter):
        x=batch.text.transpose(0,1).to(torch.float32)
        x=Variable(x).to(device)
        y=(batch.label-1).to(device)
        net.optim.zero_grad()    
        y_hat = net.forward(x, len(x))

        loss = net.criterion(y_hat, y)
        loss_val=loss.item()

        y_hat = np.argmax(y_hat.cpu().detach().numpy(),axis=1)
        train_num+=len(np.where((y_hat-y.cpu().detach().numpy())==0)[0])

        
        ej += 1
        if (ej+1)%100 == 0:
            print('epoch:', i+1,'/',epoch, ' | batch' , j*batch_size,'/',total_train ,' | loss = ', loss_val)
        loss.backward(retain_graph=True)
        net.optim.step()
        
        if ej%eval_time == 0:    
            # 测试
            with torch.no_grad():
                print('testing (epoch:',i+1,')')
                batch=test_batch
                x=batch.text.transpose(0,1).to(torch.float32)
                x=Variable(x).to(device)
                y=batch.label-1
                x=x.to(device)
                y=y.to(device)
                y_hat = net.forward(x, len(x))
                y_hat = np.argmax(y_hat.cpu().numpy(),axis=1)
                print(len(np.where((0-y_hat)==0)[0]))
                print(len(np.where((1-y_hat)==0)[0]))
                print(len(np.where((2-y_hat)==0)[0]))
                print(len(np.where((3-y_hat)==0)[0]))
                print(len(np.where((4-y_hat)==0)[0]))
                num=len(np.where((y_hat-y.cpu().numpy())==0)[0])
                print( num,total_test)
                acc = round(num/total_test, 4)
                if acc > max_acc:
                    max_acc = acc
                    if max_acc>0.5:
                        filename='res/'+info_str+'_loss_'+str(round(loss.item(),4))+'_acc_'+str(max_acc)+'.pth'
                        torch.save(net, filename)
                        print("save in " + filename)    


                print('epoch:', i+1, ' | accuracy = ', acc, ' | max_acc = ', max_acc)
    acc_list.append(acc)    
    loss_list.append(loss_val)
    acc_train = round(train_num/total_train, 4)
    with torch.no_grad():
                print('validation (epoch:',i+1,')')
                batch=test_batch
                x=batch.text.transpose(0,1).to(torch.float32)
                x=Variable(x).to(device)
                y=batch.label-1
                x=x.to(device)
                y=y.to(device)
                y_hat = net.forward(x, len(x))
                y_hat = np.argmax(y_hat.cpu().numpy(),axis=1)
                num=len(np.where((y_hat-y.cpu().numpy())==0)[0])
                print( num,total_test)
                val_acc = round(num/total_test, 4)
                if val_acc>val_max_acc:
                    val_max_acc=val_acc
                    val_max_num=0
                else:
                    val_max_num+=1
    print('train acc',acc_train)
    print('val acc',val_acc)
    if (i+1)%10==0:
                        filename='res/'+info_str+'_epoch_'+str(i+1)+'_loss_'+str(round(loss.item(),4))+'_acc_'+str(acc)+'.pth'
                        torch.save(net, filename)
                        print("save in " + filename)    



info_str+='_loss_'+str(round(loss.item(),4))+'_acc_'+str(max_acc)
from plot import plot_loss_and_acc

plot_loss_and_acc({'res': [loss_list, acc_list]},show=False,title=info_str)