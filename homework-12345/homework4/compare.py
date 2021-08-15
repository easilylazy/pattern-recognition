import os
import pandas as pd


filePath = 'csv'
savepath= 'res/'
filename='act_com_'
show=False

datas=[]
types=[]
for i,j,k in os.walk(filePath):
    print(i)
    for name in k:
        if(name.startswith('conv_act')):
            print(name)
            print('csv/'+name)
            data=pd.read_csv('csv/'+name)
            datas.append(data)
            # print(data)
            print(name.split('_')[2])
            types.append(name.split('_')[2])


# plot
import matplotlib.pyplot as plt
fig = plt.figure()
maxEpoch = 20
# stride = np.ceil(maxEpoch / 10)

# maxLoss = max([max(x[0]) for x in loss_and_acc_dict.values()]) + 0.1
# minLoss = max(0, min([min(x[0]) for x in loss_and_acc_dict.values()]) - 0.1)
minLoss=0
maxLoss=1#0.15

attributes=['test_loss',  'train_loss']
for i in range(len(types)):
    for attribute in attributes:
        plt.plot(range(1, 1 + maxEpoch), datas[i][attribute], '-s', label=types[i]+' '+attribute.split('_')[0])

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
# plt.xticks(range(0, maxEpoch + 1, 2))
# plt.axis([0, maxEpoch, minLoss, maxLoss])
plt.savefig(savepath+filename+'_loss'+'.png')
if show:
    plt.show()
fig = plt.figure()

attributes=[ 'test_acc',  'train_acc']
for i in range(len(types)):
    for attribute in attributes:
        plt.plot(range(1, 1 + maxEpoch), datas[i][attribute], '-s',  label=types[i]+' '+attribute.split('_')[0])


# maxAcc = min(1, max([max(x[1]) for x in loss_and_acc_dict.values()]) + 0.1)
# minAcc = max(0, min([min(x[1]) for x in loss_and_acc_dict.values()]) - 0.1)


# for name, lossAndAcc in loss_and_acc_dict.items():
#     plt.plot(range(1, 1 + maxEpoch), lossAndAcc[1], '-s', label=name)

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
# plt.xticks(range(0, maxEpoch + 1, 2))
# plt.axis([0, maxEpoch, minAcc, maxAcc])
plt.legend()
plt.savefig(savepath+filename+'_acc'+'.png')
if show:
    plt.show()
