import numpy as np
from numpy.core.fromnumeric import size
import torch
from torch.nn.modules.dropout import Dropout
# shape-(batch_size, channels, height, width)
m = torch.nn.Dropout(p=0.2)
input = torch.randn(20, 16)
output = m(input)
child_shape=(4,3)
batch=2
p=0.3
what=np.random.choice(2,size=child_shape,p=(0.3,1-p))
# what=np.random.randint(low=0,high=2,size=child_shape)
final=np.tile(what,(2,1,1))


print(what)
print(final)
from layers import DropoutLayer

input=np.random.random((4,5,2))
test=DropoutLayer().forward(input)
print(input)
print(test)




# 想法：
# 按照p随机产生0和1与相应位置相乘即可

# torch.nn.loss
# optimizer = torch.optim.Adagrad()
# optimizer = torch.optim.RMSprop()
# optimizer = torch.optim.
# Adam()
# adam(net.parameters(), lr=learning_rate)

#         torch.nn.Dropout(drop_out) 
