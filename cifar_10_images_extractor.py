# *-* coding: utf-8 *-*

import os
import scipy.misc

# CIFAR Images Extractor
# Python code for extracting CIFAR dataset images.

# The CIFAR dataset:
# https://www.cs.toronto.edu/~kriz/cifar.html

# Repository:
# https://github.com/amir-saniyan/CIFARImagesExtractor


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def save_batch(batch, path):
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for i in range(len(batch[b'labels'])):
        label = classes[batch[b'labels'][i]]
        image = batch[b'data'][i].reshape(3, 32, 32).transpose([1, 2, 0])
        file_name = batch[b'filenames'][i].decode('utf-8')
        directory_name = path + '/' + label
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
        full_file_name = directory_name + '/' + file_name
        print('Saving', full_file_name, '...')
        scipy.misc.imsave(full_file_name, image)


batch_1 = unpickle('./cifar-10-batches-py/data_batch_1')
save_batch(batch_1, './cifar-10-images/train')

batch_2 = unpickle('./cifar-10-batches-py/data_batch_2')
save_batch(batch_2, './cifar-10-images/train')

batch_3 = unpickle('./cifar-10-batches-py/data_batch_3')
save_batch(batch_3, './cifar-10-images/train')

batch_4 = unpickle('./cifar-10-batches-py/data_batch_4')
save_batch(batch_4, './cifar-10-images/train')

batch_5 = unpickle('./cifar-10-batches-py/data_batch_5')
save_batch(batch_5, './cifar-10-images/train')

test_batch = unpickle('./cifar-10-batches-py/test_batch')
save_batch(test_batch, './cifar-10-images/test')

print('OK')
