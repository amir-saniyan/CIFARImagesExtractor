# *-* coding: utf-8 *-*

# CIFAR Images Extractor
# Python code for extracting CIFAR dataset images.

# The CIFAR dataset:
# https://www.cs.toronto.edu/~kriz/cifar.html

# Repository:
# https://github.com/amir-saniyan/CIFARImagesExtractor

import os
import scipy.misc


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def save_batch(batch, path):
    superclasses = ['aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit and vegetables',
                    'household electrical devices', 'household furniture', 'insects', 'large carnivores',
                    'large man-made outdoor things', 'large natural outdoor scenes', 'large omnivores and herbivores',
                    'medium-sized mammals', 'non-insect invertebrates', 'people', 'reptiles', 'small mammals', 'trees',
                    'vehicles 1', 'vehicles 2']

    classes = ['apples', 'aquarium fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottles',
               'bowls', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'cans', 'castle', 'caterpillar', 'cattle',
               'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cups', 'dinosaur',
               'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo',
               'computer keyboard', 'lamp', 'lawn-mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple',
               'motorcycle', 'mountain', 'mouse', 'mushrooms', 'oak', 'oranges', 'orchids', 'otter', 'palm', 'pears',
               'pickup truck', 'pine', 'plain', 'plates', 'poppies', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray',
               'road', 'rocket', 'roses', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
               'spider', 'squirrel', 'streetcar', 'sunflowers', 'sweet peppers', 'table', 'tank', 'telephone',
               'television', 'tiger', 'tractor', 'train', 'trout', 'tulips', 'turtle', 'wardrobe', 'whale', 'willow',
               'wolf', 'woman', 'worm']

    for i in range(len(batch[b'filenames'])):
        coarse_label = superclasses[batch[b'coarse_labels'][i]]
        fine_label = classes[batch[b'fine_labels'][i]]
        image = batch[b'data'][i].reshape(3, 32, 32).transpose([1, 2, 0])
        file_name = batch[b'filenames'][i].decode('utf-8')
        directory_name = path + '/' + coarse_label + '/' + fine_label
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
        full_file_name = directory_name + '/' + file_name
        print('Saving', full_file_name, '...')
        scipy.misc.imsave(full_file_name, image)


train = unpickle('./cifar-100-python/train')
save_batch(train, './cifar-100-images/train')

test = unpickle('./cifar-100-python/test')
save_batch(test, './cifar-100-images/test')

print('OK')
