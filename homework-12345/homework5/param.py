import getopt,sys

max_len = 65 #句子最大长度
embedding_size = 300
hidden_size = 100
batch_size = 64
epoch = 100
label_num = 6
eval_time = 100 # 每训练100个batch后对测试集或验证集进行测试
learning_rate_SGD=0.01
weight_decay=0.01
try:
    argv=(sys.argv[1:])
    opts, args = getopt.getopt(argv,"b:e:m:i:l:w:d:t:h:",["ifile=","ofile="])
except getopt.GetoptError:
    print ('test.py -b <batch_size> -m <max len> -e <epoch> -i <init_std> -l <learning rate> -w <weight decay> -t <whether test>')
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        hidden_size=eval(arg)
    elif opt == '-b':
        batch_size=eval(arg)
    elif opt == '-d':
        eval_time=eval(arg)
    elif opt == '-e':
        epoch=eval(arg)
    elif opt == '-m':
        max_len=eval(arg)
    elif opt == '-i':
        init_std=eval(arg)
    elif opt == '-l':
        learning_rate_SGD=eval(arg)
    elif opt == '-w':
        weight_decay=eval(arg)
    elif opt == '-t':
        if arg == 'False':
            test_choice=False
info_str= (
    "_lr_"
    + str(learning_rate_SGD)
    + "_de_"
    + str(weight_decay)
    + '_len_'
    +str(max_len)
    +'_hid_'
    +str(hidden_size)
    +'_emb_'
    +str(embedding_size)
    + "_epo_"
    + str(epoch)
    + "_bat_"
    + str(batch_size)
    +'_eval_'
    +str(eval_time)
)
print(info_str)

def get_param():

    return info_str,max_len ,embedding_size ,hidden_size ,batch_size,epoch,label_num ,eval_time,learning_rate_SGD,weight_decay 
if __name__=='__main__':
    print(get_param())