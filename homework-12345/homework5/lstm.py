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
class Config(object):
    '''
    全局配置参数
    '''
    status = 'train' # 执行 train_eval or test, 默认执行train_eval
    use_model = 'TextCNN' # 使用何种模型, 默认使用TextCNN
    output_folder = 'output_data/'  # 已处理的数据所在文件夹
    data_name = 'SST-2' # SST-1(fine-grained) SST-2(binary)
    SST_path  = 'data/stanfordSentimentTreebank/' # 数据集所在路径
    emb_file = 'data/glove.6B.300d.txt' # 预训练词向量所在路径
    emb_format = 'glove' # embedding format: word2vec/glove
    min_word_freq = 1 # 最小词频
    max_len = 40 # 采样最大长度
        # 模型参数
    model_name = 'TextAttnBiLSTM' # 模型名
    class_num = 5 if data_name == 'SST-1' else 2 # 分类类别
    class_num=5

    embed_dropout = 0.3 # dropout
    model_dropout = 0.5 # dropout
    fc_dropout = 0.5 # dropout
    num_layers = 2 # LSTM层数
    embed_dim = 128 # 未使用预训练词向量的默认值
    use_embed = True # 是否使用预训练
    use_gru = False # 是否使用GRU
    grad_clip = 4. # 梯度裁剪阈值
class ModelConfig():
    '''
    模型配置参数
    '''
    # 全局配置参数
    opt = Config()

    # 数据参数
    output_folder = opt.output_folder
    data_name = opt.data_name
    SST_path  = opt.SST_path
    emb_file = opt.emb_file
    emb_format = opt.emb_format
    output_folder = opt.output_folder
    min_word_freq = opt.min_word_freq
    max_len = opt.max_len

    # 训练参数
    epochs = 120  # epoch数目，除非early stopping, 先开20个epoch不微调,再开多点epoch微调
    batch_size = 16 # batch_size
    workers = 4  # 多处理器加载数据
    lr = 1e-4  # 如果要微调时，学习率要小于1e-3,因为已经是很优化的了，不用这么大的学习率
    weight_decay = 1e-5 # 权重衰减率
    decay_epoch = 15 # 多少个epoch后执行学习率衰减
    improvement_epoch = 30 # 多少个epoch后执行early stopping
    is_Linux = False # 如果是Linux则设置为True,否则设置为else, 用于判断是否多处理器加载
    print_freq = 100  # 每隔print_freq个iteration打印状态
    checkpoint =  None  # 模型断点所在位置, 无则None
    best_model = None # 最优模型所在位置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型参数
    model_name = 'TextAttnBiLSTM' # 模型名
    class_num = 5 if data_name == 'SST-1' else 2 # 分类类别
    embed_dropout = 0.3 # dropout
    model_dropout = 0.5 # dropout
    fc_dropout = 0.5 # dropout
    num_layers = 2 # LSTM层数
    embed_dim = 128 # 未使用预训练词向量的默认值
    use_embed = True # 是否使用预训练
    use_gru = True # 是否使用GRU
    grad_clip = 4. # 梯度裁剪阈值

class Attn(nn.Module):
    '''
    Attention Layer
    '''
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.attn = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        '''
        :param x: (batch_size, max_len, hidden_size)
        :return alpha: (batch_size, max_len)
        '''
        x = torch.tanh(x) # (batch_size, max_len, hidden_size)
        x = self.attn(x).squeeze(2) # (batch_size, max_len)
        alpha = F.softmax(x, dim=1).unsqueeze(1) # (batch_size, 1, max_len)
        return alpha

class ModelAttnBiLSTM(nn.Module):
    '''
    BiLSTM: BiLSTM, BiGRU
    '''
    def __init__(self, vocab_size, embed_dim, hidden_size, pretrain_embed, use_gru, embed_dropout, fc_dropout, model_dropout, num_layers, class_num, use_embed):

        super(ModelAttnBiLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        
        if use_embed:
            self.embedding = nn.Embedding(vocab_size, embed_dim).from_pretrained(pretrain_embed, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.embed_dropout = nn.Dropout(embed_dropout)   
         
        if use_gru:
            self.bilstm = nn.GRU(embed_dim, hidden_size, num_layers, dropout=(0 if num_layers == 1 else model_dropout), bidirectional=True, batch_first=True)
        else:
            self.bilstm = nn.LSTM(embed_dim, hidden_size, num_layers, dropout=(0 if num_layers == 1 else model_dropout), bidirectional=True, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, class_num)
        
        self.fc_dropout = nn.Dropout(fc_dropout) 

        self.attn = Attn(hidden_size)
        self.criterion  = nn.CrossEntropyLoss().to(device)
        self.optim = torch.optim.Adamax(self.parameters(),lr=learning_rate,weight_decay=0)
    def forward(self, x):
        '''
        :param x: [batch_size, max_len]
        :return logits: logits
        '''
        import pdb
        # pdb.set_trace()
        x = self.embedding(x) # (batch_size, max_len, word_vec)
        x = self.embed_dropout(x)
        # 输入的x是所有time step的输入, 输出的y实际每个time step的hidden输出
        # _是最后一个time step的hidden输出
        # 因为双向,y的shape为(batch_size, max_len, hidden_size*num_directions), 其中[:,:,:hidden_size]是前向的结果,[:,:,hidden_size:]是后向的结果
        y, _ = self.bilstm(x) # (batch_size, max_len, hidden_size*num_directions)
        y = y[:,:,:self.hidden_size] + y[:,:,self.hidden_size:] # (batch_size, max_len, hidden_size)
        alpha = self.attn(y) # (batch_size, 1, max_len)
        r = alpha.bmm(y).squeeze(1) # (batch_size, hidden_size)
        h = torch.tanh(r) # (batch_size, hidden_size)
        logits = self.fc(h) # (batch_size, class_num)
        logits = self.fc_dropout(logits)
        return logits

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
# train_, test_, vocab = processData()
# embedding_table = word_embedding(len(vocab), embedding_size)
train_iter, val_iter, _ = data.BucketIterator.splits(
    (train, val, test), batch_size=batch_size,shuffle=True)
test_batch = next(iter(test_iter)) # for batch in train_iter
val_batch = next(iter(val_iter)) # for batch in train_iter

# %%
opt=Config()
# print batch information
net = ModelAttnBiLSTM(vocab_size=len(TEXT.vocab), 
                      embed_dim=300, 
                      hidden_size=300,
                      class_num=opt.class_num,
                      pretrain_embed=pretrained_embeddings,
                      num_layers=opt.num_layers,
                      model_dropout=opt.model_dropout, 
                      fc_dropout=opt.fc_dropout,
                      embed_dropout=opt.embed_dropout,
                      use_gru=False,#opt.use_gru, 
                      use_embed=True,)#opt.use_embed)
net=net.to(device)
# net.embedding_table.weight.data.copy_(pretrained_embeddings)
# embedding_table)
print(net.embedding)
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
        x=batch.text.transpose(0,1)#.to(torch.float32)
        x=Variable(x).to(device)
        y=(batch.label-1).to(device)
        net.optim.zero_grad()    
        y_hat = net.forward(x)
        import pdb
        # pdb.set_trace()
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
                x=batch.text.transpose(0,1)#.to(torch.float32)
                x=Variable(x).to(device)
                y=batch.label-1
                x=x.to(device)
                y=y.to(device)
                y_hat = net.forward(x)
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
                x=batch.text.transpose(0,1)#.to(torch.float32)
                x=Variable(x).to(device)
                y=batch.label-1
                x=x.to(device)
                y=y.to(device)
                y_hat = net.forward(x)
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