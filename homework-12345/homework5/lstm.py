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

# # print information about the data
# print('train.fields', train.fields)
# print('len(train)', len(train))
# print('vars(train[0])', vars(train[0]))


# %%

# build the vocabulary
# you can use other pretrained vectors, refer to https://github.com/pytorch/text/blob/master/torchtext/vocab.py
TEXT.build_vocab(train, vectors=Vectors(name='vector.txt', cache='./data'))
LABEL.build_vocab(train)
# # We can also see the vocabulary directly using either the stoi (string to int) or itos (int to string) method.
# print(TEXT.vocab.itos[:10])
# print(LABEL.vocab.stoi)
# print(TEXT.vocab.freqs.most_common(20))




# %%


################################
# After build your network 
################################


# Copy the pre-trained word embeddings we loaded earlier into the embedding layer of our model.
pretrained_embeddings = TEXT.vocab.vectors

print(pretrained_embeddings.shape)


# %%



class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


# %%
EMBEDDING_DIM=300
HIDDEN_DIM=10


# %%
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM,len(TEXT.vocab), len(LABEL.vocab))
# model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM,len(word_to_ix), len(tag_to_ix))

# you should maintain a nn.embedding layer in your network
model.word_embeddings.weight.data.copy_(pretrained_embeddings)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()


# %%

# make iterator for splits
train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (train, val, test), batch_size=64)

# print batch information
batch = next(iter(train_iter)) # for batch in train_iter
# Attention: batch.label in the range [1,5] not [0,4] !!!


# %%

batch_size=64

# %%
from torch.autograd import Variable
for epoch in range(100):  # again, normally you would NOT do 300 epochs, it is toy data
    batch = next(iter(train_iter)) # for batch in train_iter
    tag_scores=torch.zeros((batch_size,6))
    for i in range(batch_size):
    # for sentence, tags in batch:
        sentence=batch.text[:,i]
        tags=batch.label[i]
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        sentence_in = sentence
        #prepare_sequence(sentence, word_to_ix)
        # targets = tags
        # prepare_sequence(tags, tag_to_ix)

        # Step 3. Run our forward pass.
        tag_scores[i]=(model(sentence_in)[-1])

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
    targets=batch.label
    tag_scores=Variable(tag_scores,requires_grad=True)
    loss = loss_function(tag_scores, targets)
    loss.backward()
    optimizer.step()
    print(loss)

import pdb
pdb.set_trace()

# # %%

# # See what the scores are after training
# with torch.no_grad():
#     inputs = prepare_sequence(training_data[0][0], word_to_ix)
#     tag_scores = model(inputs)

#     # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
#     # for word i. The predicted tag is the maximum scoring tag.
#     # Here, we can see the predicted sequence below is 0 1 2 0 1
#     # since 0 is index of the maximum value of row 1,
#     # 1 is the index of maximum value of row 2, etc.
#     # Which is DET NOUN VERB DET NOUN, the correct sequence!
#     print(tag_scores)


