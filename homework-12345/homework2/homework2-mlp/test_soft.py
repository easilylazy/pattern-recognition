# from criterion import SoftmaxCrossEntropyLossLayer
# import tensorflow.compat.v1 as tf
# import numpy
# import matplotlib.pyplot as plt
# #  from tf.losses import softmax_cross_entropy
# test_labels=[1,2,3,]
# print(test_labels)
# adata=numpy.array(test_labels)
# print(adata)
# def make_one_hot(data1):
#     return (numpy.arange(10)==data1[:,None]).astype(numpy.integer)

# my_one_hot =make_one_hot(adata)
# sys_=[]
# own_=[]
# ratio=[]
# for i in range(10):
#     logits=numpy.random.random(my_one_hot.shape)
#     # print(my_one_hot)
#     img=tf.losses.softmax_cross_entropy(my_one_hot,logits)
#     print(tf.losses.softmax_cross_entropy(my_one_hot,logits))
#     # criterion2 = SoftmaxCrossEntropyLossLayer()
#     sess=tf.Session()
#     #sess.run(tf.initialize_all_variables())
#     sess.run(tf.global_variables_initializer())
#     print("out1=",type(img))
#     #转化为numpy数组
#     #通过.eval函数可以把tensor转化为numpy类数据
#     img_numpy=img.eval(session=sess)
#     print("out2=",type(img_numpy))
#     print(img_numpy)
#     #转化为tensor
#     img_tensor= tf.convert_to_tensor(img_numpy)
#     print("out2=",type(img_tensor))


#     criterion=SoftmaxCrossEntropyLossLayer()
#     loss=criterion.forward(logits,my_one_hot)
#     print(loss)

#     print('---------'+str(img_numpy/loss))
#     sys_.append(img_numpy)
#     own_.append(loss)
#     ratio.append(img_numpy/loss)
# plt.plot(sys_)
# plt.plot(own_)
# plt.plot(ratio)
# plt.show()


import sys, getopt
from layers import ReLULayer,SigmoidLayer
from datetime import date, datetime
import numpy as np

def main3():
    re=ReLULayer()
    si=SigmoidLayer()
    t1=datetime.now()
    for i in range(100):
        h=np.random.random((500,500))
        re.forward(h)
        re.backward(h)
    print(datetime.now()-t1)
    t1=datetime.now()
    for i in range(100):
        h=np.random.random((500,500))
        si.forward(h)
        si.backward(h)
    print(datetime.now()-t1)
    t1=datetime.now()
    for i in range(100):
        h=np.random.random((500,500))
        re.forward(h)
    print(datetime.now()-t1)
    t1=datetime.now()
    for i in range(100):
        h=np.random.random((500,500))
        si.forward(h)
        si.backward(h)
    print(datetime.now()-t1)

def main2(argv):
    batch_size = 100
    max_epoch = 10
    init_std = 0.01

    learning_rate_SGD = 0.001
    weight_decay = 0.1

    hidden_layer1=128
    hidden_layer2=64
    disp_freq = 50

    test_choice = True
    try:
        argv=(sys.argv[1:])
        opts, args = getopt.getopt(argv,"hb:m:i:l:w:d:t:1:2:",["ifile=","ofile="])
    except getopt.GetoptError:
        print ('test.py -b <batch_size> -m <max epoch> -i <init_std> -l <learning rate> -w <weight decay> -t <whether test>')
        sys.exit(2)
    print(opts)
    for opt, arg in opts:
        if opt == '-h':
            print ('test.py -b <batch_size> -m <max epoch> -i <init_std> -l <learning rate> -w <weight decay> -t <whether test>')
            sys.exit()
        elif opt == '-b':
            batch_size=eval(arg)
        elif opt == '-m':
            max_epoch=eval(arg)
        elif opt == '-i':
            init_std=eval(arg)
        elif opt == '-l':
            learning_rate_SGD=eval(arg)
        elif opt == '-w':
            weight_decay=eval(arg)
        elif opt == '-t':
            if arg == 'False':
                test_choice=False
        elif opt == '-1':
            hidden_layer1=eval(arg)
        elif opt == '-2':
            hidden_layer2=eval(arg)
    info_str= (
        "_lr_"
        + str(learning_rate_SGD)
        + "_de_"
        + str(weight_decay)
        + "_epo_"
        + str(max_epoch)
        + "_bat_"
        + str(batch_size)
        + "_h1_"
        + str(hidden_layer1)
        + "_h2_"
        + str(hidden_layer2)
    )
    print(info_str)
    # %% [markdown]
    # ## 1. MLP with Euclidean Loss
    # In part-1, you need to train a MLP with **Euclidean Loss**.  
    # **Sigmoid Activation Function** and **ReLU Activation Function** will be used respectively.
    # ### TODO
    # Before executing the following code, you should complete **./optimizer.py** and **criterion/euclidean_loss.py**.

    print(h1,h2)
    print(info_str)


if __name__ == "__main__":
    main3()
#    main2(sys.argv[1:])