""" Plot loss and accuracy curve """

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_loss_and_acc(loss_and_acc_dict,save=True,show=True,path='res/',title='lr_0.01'):
	try:
		title='up_'+title
		if os.path.exists(path) is False:
			os.mkdir(path)
		fig = plt.figure()
		tmp = list(loss_and_acc_dict.values())
		maxEpoch = len(tmp[0][0])
		stride = np.ceil(maxEpoch / 10)

		maxLoss = max([max(x[0]) for x in loss_and_acc_dict.values()]) + 0.1
		minLoss = max(0, min([min(x[0]) for x in loss_and_acc_dict.values()]) - 0.1)

		for name, lossAndAcc in loss_and_acc_dict.items():
			plt.plot(range(1, 1 + maxEpoch), lossAndAcc[0], '-s', label=name)

		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		plt.legend()
		plt.xticks(range(0, maxEpoch + 1, 2))
		plt.axis([0, maxEpoch, minLoss, maxLoss])

		if save:
			plt.savefig(path+'loss_'+title+'.png')
		if show:
			plt.show()

		maxEpoch = len(tmp[0][1])

		maxAcc = min(1, max([max(x[1]) for x in loss_and_acc_dict.values()]) + 0.1)
		minAcc = max(0, min([min(x[1]) for x in loss_and_acc_dict.values()]) - 0.1)



		fig = plt.figure()

		for name, lossAndAcc in loss_and_acc_dict.items():
			import pdb
			pdb.set_trace()
			plt.plot(range(1, 1 + maxEpoch), lossAndAcc[1], '-s', label=name)

		plt.xlabel('Epoch')
		plt.ylabel('Accuracy')
		plt.xticks(range(0, maxEpoch + 1, 2))
		plt.axis([0, maxEpoch, minAcc, maxAcc])
		plt.legend()
		if save:
			plt.savefig(path+'acc_'+title+'.png')
		if show:
			plt.show()
	except:
		pass


if __name__ == '__main__':
	loss = [x for x in range(10, 0, -1)]
	acc = [x / 10. for x in range(0, 10)]
	plot_loss_and_acc({'as': [loss, acc]})
