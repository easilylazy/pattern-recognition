# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'
# %%
from datetime import date, time
import os
import numpy


# %%
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

# %%
class DataSet(object):
    def __init__(self, images, labels, one_hot=False, dtype=numpy.float32):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        """
        assert (
            len(images) == len(labels)
        ), "images.shape: %s labels.shape: %s" % (len(images), len(labels))
        self._num_examples = len(images)
        # shape is [num examples, rows*columns*channels] 
        images = numpy.multiply(images, 1.0 / 255.0)
        if one_hot == True:
            labels=dense_to_one_hot(labels,num_classes=10)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def __getitem__(self, index):
        return self._images[index], self._labels[index]

    def __len__(self):
        return len(self._images)

    def next_batch(self, batch_size, fake_data=False):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


# %%
def load_data(path,one_hot=False):
    class DataSets(object):
        pass
    data_sets = DataSets()
    # meta data
    metadata=unpickle(os.path.join(path,'batches.meta'))
    num_cases_per_batch=(metadata[b'num_cases_per_batch'])
    num_labels=len(metadata[b'label_names'])
    num_pixels=(metadata[b'num_vis'])
    # data -- a 10000x3072 numpy array of uint8s; 0-255
    train_images=[]
    # labels -- a list of 10000 numbers in the range 0-9.
    train_labels=[]
    # load train 
    for i in range(5):
        batch_file='data_batch_'+str(i+1)
        data_dict=unpickle(os.path.join(path,batch_file))
        train_images.extend(data_dict[b'data'])
        train_labels.extend(data_dict[b'labels'])
    # load test
    batch_file='test_batch'
    data_dict=unpickle(os.path.join(path,batch_file))
    test_images=(data_dict[b'data'])
    test_labels=(data_dict[b'labels'])
    
    data_sets.train = DataSet(numpy.array(train_images), numpy.array(train_labels), one_hot=one_hot)
    data_sets.test = DataSet(numpy.array(test_images), numpy.array(test_labels), one_hot=one_hot)
    return data_sets



# %%

if __name__=='__main__':
    from datetime import datetime
    start=datetime.now()
    path='data/cifar-10-batches-py'
    data_sets=load_data(path=path,one_hot=True)
    print('time cost: ',datetime.now()-start)
    print('num of train items: ',data_sets.train.num_examples)
    print('example of train label: ',data_sets.train.labels[0])
    X0=data_sets.train.images[0]
    X = X0.reshape(3, 32, 32)
    print(numpy.sum(X[1][1][:10]-X0[32*32+32:32*32+32+10]))

