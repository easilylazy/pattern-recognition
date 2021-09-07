import os
import pandas as pd


filePath = 'csv'
savepath= 'img/'
filename='unit_com_'
# show=True
show=True

datas=[]
types=[]
for i,j,k in os.walk(filePath):
    print(i)
    for name in k:
        if(name.startswith('dp_othernewstd')):
            print(name)
            print('csv/'+name)
            data=pd.read_csv('csv/'+name)
            datas.append(data)
            # print(data)
            print(name.split('_')[11])
            print(name.split('_')[1]+' batch-'+name.split('_')[11])
            types.append(name.split('_')[2]+' '+name.split('_')[5])


# plot
import matplotlib.pyplot as plt
fig = plt.figure()
maxEpoch = 164
# strides=40
strides = int(maxEpoch // 20)

# maxLoss = max([max(x[0]) for x in loss_and_acc_dict.values()]) + 0.1
# minLoss = max(0, min([min(x[0]) for x in loss_and_acc_dict.values()]) - 0.1)
minLoss=0
maxLoss=1#0.15

attributes=['test_loss',  'train_loss']
for i in range(len(types)):
    for attribute in attributes:
        if(len(datas[i][attribute])==328):
            plt.plot(range(1, 1 + maxEpoch,strides), datas[i][attribute][::strides*2], '-s', label=types[i]+' '+attribute.split('_')[0])
        else:
            plt.plot(range(1, 1 + maxEpoch,strides), datas[i][attribute][::strides], '-s', label=types[i]+' '+attribute.split('_')[0])

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.xticks(range(0, maxEpoch + 1, strides))
# plt.axis([0, maxEpoch, minLoss, maxLoss])
plt.savefig(savepath+filename+'_loss'+'.png')
if show:
    plt.show()
fig = plt.figure()

attributes=[ 'test_acc',  'train_acc']
for i in range(len(types)):
    for attribute in attributes:
        if(len(datas[i][attribute])==328):
            plt.plot(range(1, 1 + maxEpoch,strides), datas[i][attribute][::strides*2], '-s', label=types[i]+' '+attribute.split('_')[0])
        else:
            plt.plot(range(1, 1 + maxEpoch,strides), datas[i][attribute][::strides], '-s', label=types[i]+' '+attribute.split('_')[0])


# maxAcc = min(1, max([max(x[1]) for x in loss_and_acc_dict.values()]) + 0.1)
# minAcc = max(0, min([min(x[1]) for x in loss_and_acc_dict.values()]) - 0.1)


# for name, lossAndAcc in loss_and_acc_dict.items():
#     plt.plot(range(1, 1 + maxEpoch), lossAndAcc[1], '-s', label=name)

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks(range(0, maxEpoch + 1, strides))
# plt.axis([0, maxEpoch, minAcc, maxAcc])
plt.legend()
plt.savefig(savepath+filename+'_acc'+'.png')
if show:
    plt.show()
