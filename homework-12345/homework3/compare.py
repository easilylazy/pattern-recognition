import os
import pandas as pd


filePath = 'csv'
savepath= 'res/'
filename='dropout_site_'
show=True
# False

datas=[]
types=[]
# parameter
# for i,j,k in os.walk(filePath):
#     print(i)
#     for name in k:
#         if(name.startswith('dropout')):
#             splits=name.split('_')
#             if (splits[1])=='fc1':# and eval(splits[7])==0.005 and eval(splits[9])==0.2:
             
#                 print(name)
#                 print('csv/'+name)
#                 data=pd.read_csv('csv/'+name)
#                 datas.append(data)
#                 # print(data)
#                 print()
#                 types.append(name.split('_')[1]+' decay '+splits[7]+' drop '+splits[9])
#         if(name.startswith('normal')):
#             data=pd.read_csv('csv/'+name)
#             types.append(name.split('_')[0])

#             datas.append(data)
# different place
for i,j,k in os.walk(filePath):
    print(i)
    for name in k:
        if(name.startswith('dropout')):
            splits=name.split('_')
            if eval(splits[5])==0.001 and eval(splits[7])==0.005 and eval(splits[9])==0.2:
             
                print(name)
                print('csv/'+name)
                data=pd.read_csv('csv/'+name)
                datas.append(data)
                # print(data)
                print()
                types.append(name.split('_')[1])
        if(name.startswith('normal')):
            data=pd.read_csv('csv/'+name)
            types.append(name.split('_')[0])

            datas.append(data)
    


# plot
import matplotlib.pyplot as plt
fig = plt.figure()
maxEpoch = 20
# stride = np.ceil(maxEpoch / 10)

# maxLoss = max([max(x[0]) for x in loss_and_acc_dict.values()]) + 0.1
# minLoss = max(0, min([min(x[0]) for x in loss_and_acc_dict.values()]) - 0.1)
minLoss=0
maxLoss=1#0.15

attributes=['train_loss']
for i in range(len(types)):
    for attribute in attributes:
        plt.plot(range(1, 1 + maxEpoch), datas[i][attribute], '-s', label=types[i])

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
# plt.xticks(range(0, maxEpoch + 1, 2))
# plt.axis([0, maxEpoch, minLoss, maxLoss])
plt.savefig(savepath+filename+'_loss'+'.png')
if show:
    plt.show()
fig = plt.figure()

attributes=[  'train_acc']
for i in range(len(types)):
    for attribute in attributes:
        plt.plot(range(1, 1 + maxEpoch), datas[i][attribute], '-s',  label=types[i])


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
