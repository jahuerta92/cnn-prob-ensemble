import numpy as np
import pandas as pd
import scipy as sp

from skimage.feature import greycomatrix, greycoprops

import tensorflow as tf
import tensorflow_datasets as tfds

GREY_LEVELS = 256
AUX_MAT = np.zeros(shape=(GREY_LEVELS, GREY_LEVELS))
for i in range(GREY_LEVELS):
    for j in range(GREY_LEVELS):
        AUX_MAT[i, j] = i + j


def variance(glcm):
    l_i, l_j, dist, theta = glcm.shape

    var_list = list()
    arr_i = np.array(range(0, l_i))
    for t in range(theta):
        for d in range(dist):
            p = glcm[:, :, d, t]
            mn = np.mean(p)
            var = np.sum((arr_i - mn) * np.sum(p, axis=1))
            var_list.append(var)

    return np.mean(var_list)


def mean_tex(glcm):
    l_i, l_j, dist, theta = glcm.shape

    mt_list = list()
    for t in range(theta):
        for d in range(dist):
            p = glcm[:, :, d, t]
            mean_tex = np.sum(AUX_MAT * p)
            mt_list.append(mean_tex)

    return np.mean(mt_list)


def entropy(glcm):
    l_i, l_j, dist, theta = glcm.shape

    ent_list = list()
    for t in range(theta):
        for d in range(dist):
            p = glcm[:, :, d, t]
            log = np.where(p != 0, np.log(p), 0)
            ent = np.sum(p * log)
            ent_list.append(ent)

    return np.mean(ent_list)


def extract_features(image, label, index=0, verbose=False):
    '''Example use:
        import tensorflow as tf
        import tensorflow_datasets as tfds

        import pandas as pd

        data, info = tfds.load('malaria', split='train', as_supervised=True, shuffle_files=False, with_info=True)

        feature_list, labels = list(), list()
        for i, (image, label) in enumerate(tfds.as_numpy(data)):
          feature_list.append(extract_features(image, i))
          labels.append(label)

        feature_set = pd.DataFrame(feature_list)
        feature_set['label'] = labels
        feature_set.to_csv('malaria_features.csv')
    '''
    # image input shape --> (width, height, colour)
    w, h, col = image.shape
    nop = w * h
    feature_dict = {'red': dict(),
                    'green': dict(),
                    'blue': dict()}

    # Get features from individual colour channels
    for i, key in zip(range(col), feature_dict.keys()):
        # Extract a colour channel (r-->g-->b)
        channel = image[:, :, i]

        # Simple spectral features
        feature_dict[key]['mean'] = np.mean(channel)
        feature_dict[key]['std'] = np.std(channel)
        feature_dict[key]['skew'] = sp.stats.skew(channel.flatten())

        # Histogram spectral features
        channel_hist = np.histogram(channel, bins=5)[0]

        feature_dict[key]['hist-low'] = channel_hist[0] / nop
        feature_dict[key]['hist-medium-low'] = channel_hist[1] / nop
        feature_dict[key]['hist-medium'] = channel_hist[2] / nop
        feature_dict[key]['hist-medium-high'] = channel_hist[3] / nop
        feature_dict[key]['hist-high'] = channel_hist[4] / nop
        # GLCM textural features
        channel_glcm = greycomatrix(channel, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                                    levels=256,
                                    normed=True)

        # ‘contrast’, ‘dissimilarity’, ‘homogeneity’, ‘energy’, ‘correlation’,
        # ‘ASM’, 'entropy', 'mean-tex', 'variance'

        feature_dict[key]['contrast'] = np.mean(greycoprops(channel_glcm, 'contrast'))
        feature_dict[key]['dissimilarity'] = np.mean(greycoprops(channel_glcm, 'dissimilarity'))
        feature_dict[key]['homogeneity'] = np.mean(greycoprops(channel_glcm, 'homogeneity'))
        feature_dict[key]['correlation'] = np.mean(greycoprops(channel_glcm, 'correlation'))
        feature_dict[key]['ASM'] = np.mean(greycoprops(channel_glcm, 'ASM'))
        feature_dict[key]['entropy'] = entropy(channel_glcm)
        feature_dict[key]['mean-tex'] = mean_tex(channel_glcm)
        feature_dict[key]['variance'] = variance(channel_glcm)

    # Cross-channel features
    cross_feature_dict = {'red-green': dict(),
                          'red-blue': dict(),
                          'green-blue': dict()}

    cross_feature_dict['red-green']['difference'] = feature_dict['red']['mean'] - feature_dict['green']['mean']
    cross_feature_dict['red-green']['ratio'] = feature_dict['red']['mean'] / feature_dict['green']['mean']
    cross_feature_dict['red-blue']['difference'] = feature_dict['red']['mean'] - feature_dict['blue']['mean']
    cross_feature_dict['red-blue']['ratio'] = feature_dict['red']['mean'] / feature_dict['blue']['mean']
    cross_feature_dict['green-blue']['difference'] = feature_dict['blue']['mean'] - feature_dict['green']['mean']
    cross_feature_dict['green-blue']['ratio'] = feature_dict['blue']['mean'] / feature_dict['green']['mean']

    flat_feature_dict = {'{}_{}'.format(k1, k2): feature_dict[k1][k2]
                         for k1 in feature_dict.keys()
                         for k2 in feature_dict[k1].keys()}

    flat_feature_dict.update({'{}_{}'.format(k1, k2): cross_feature_dict[k1][k2]
                              for k1 in cross_feature_dict.keys()
                              for k2 in cross_feature_dict[k1].keys()})
    if verbose:
        print('[{}]Image processed'.format(index))

    return flat_feature_dict


def tdfs_to_features(name, dir, split=None):
    if split is not None:
        data, info = tfds.load(name, split=split, as_supervised=True, shuffle_files=False, with_info=True)
    else:
        data, info = tfds.load(name, split='train', as_supervised=True, shuffle_files=False, with_info=True)

    feature_list, labels = list(), list()
    for i, (image, label) in enumerate(tfds.as_numpy(data)):
        feature_list.append(extract_features(image, i))
        labels.append(label)

    feature_set = pd.DataFrame(feature_list)
    feature_set['label'] = labels

    if split is not None:
        feature_set.to_csv('{}/{}-{}_features.csv'.format(dir, name, split))
    else:
        feature_set.to_csv('{}/{}_features.csv'.format(dir, name))

    return True

def data_to_features(data, verbose=False):
    feature_list, labels = list(), list()
    for i, (image, label) in enumerate(tfds.as_numpy(data)):
        feature_list.append(extract_features(image, i, False))
        labels.append(label)

    feature_set = pd.DataFrame(feature_list)
    feature_set['label'] = labels
    return feature_set
