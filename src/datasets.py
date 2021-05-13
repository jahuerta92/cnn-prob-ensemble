from src.Image import extract_features
import tensorflow as tf
import tensorflow_datasets as tfds
import os
import pandas as pd
from tqdm import tqdm
import sys
from multiprocessing import Pool

IMG_SIZE = 224

TF_DATASETS_FOLDER = "tf_datasets"
INFO_DATASETS_FOLDER = "info_datasets"
FEATURES_DATASETS_FOLDER = "features_datasets"

if not os.path.exists(TF_DATASETS_FOLDER):
    os.makedirs(TF_DATASETS_FOLDER)

if not os.path.exists(INFO_DATASETS_FOLDER):
    os.makedirs(INFO_DATASETS_FOLDER)

if not os.path.exists(FEATURES_DATASETS_FOLDER):
    os.makedirs(FEATURES_DATASETS_FOLDER)


# def transform_images(row, size):
#     x_train = tf.image.resize(row['image'], (size, size))
#     x_train = x_train  / 255
#     return x_train, row['label']


def transform_images(image, size, label):
    x_train = tf.image.resize(image, (size, size))
    x_train = x_train / 255
    return (x_train, label)


def load_dataset_by_name(dataset_name, batch_size):

    if dataset_name == "malaria":
        return load_malaria_dataset(batch_size=batch_size)

    elif dataset_name == "cars196":
        return load_cars196_dataset(batch_size=batch_size)

    elif dataset_name == "caltech_birds2011":
        return load_caltech_birds2011_dataset(batch_size=batch_size)

    elif dataset_name == "cats_vs_dogs":
        return load_cats_vs_dogs_dataset(batch_size=batch_size)

    elif dataset_name == "cassava":
        return load_cassava_dataset(batch_size=batch_size)

    elif dataset_name == "plant_leaves":
        return load_plant_leaves_dataset(batch_size=batch_size)

    elif dataset_name == "deep_weeds":
        return load_deep_weeds_dataset(batch_size=batch_size)

    elif dataset_name == "oxford_flowers102":
        return load_oxford_flowers102_dataset(batch_size=batch_size)

    elif dataset_name == "citrus_leaves":
        return load_citrus_leaves_dataset(batch_size=batch_size)

    elif dataset_name == "plant_village":
        return load_plant_village_dataset(batch_size=batch_size)

    else:
        print("dataset not found!")
        sys.exit(0)


###############################################################################
# plant_village
###############################################################################
def load_plant_village_dataset(batch_size):

    train_prop = 70
    valid_prop = 10
#     test_prop=20

    split_train = "train[:{0:d}%]".format(train_prop)
    split_validation = "train[{0:d}:{1:d}%]".format(train_prop, train_prop + valid_prop)
    split_test = "train[{0:d}:{1:d}%]".format(train_prop + valid_prop, 100)

    no_labels = 38
    dataset_name = "plant_village"
    dataset_version = "1.0.2"
    print("Dataset: {}".format(dataset_name))

    dataset_code = '{0:s}:{1:s}'.format(dataset_name, dataset_version)

    data_loaded = prepare_dataset(dataset_code=dataset_code,
                                  split_train=split_train,
                                  split_validation=split_validation,
                                  split_test=split_test,
                                  batch_size=batch_size,
                                  dataset_name=dataset_name,
                                  no_labels=no_labels)

    return data_loaded


###############################################################################
# Citrus leaves
###############################################################################
def load_citrus_leaves_dataset(batch_size):

    train_prop = 70
    valid_prop = 10
#     test_prop=20

    split_train = "train[:{0:d}%]".format(train_prop)
    split_validation = "train[{0:d}:{1:d}%]".format(train_prop, train_prop + valid_prop)
    split_test = "train[{0:d}:{1:d}%]".format(train_prop + valid_prop, 100)

    no_labels = 4
    dataset_name = "citrus_leaves"
    dataset_version = "0.1.2"
    print("Dataset: {}".format(dataset_name))

    dataset_code = '{0:s}:{1:s}'.format(dataset_name, dataset_version)

    data_loaded = prepare_dataset(dataset_code=dataset_code,
                                  split_train=split_train,
                                  split_validation=split_validation,
                                  split_test=split_test,
                                  batch_size=batch_size,
                                  dataset_name=dataset_name,
                                  no_labels=no_labels)

    return data_loaded


###############################################################################
# Oxford flowers 102
###############################################################################
def load_oxford_flowers102_dataset(batch_size):

    split_train = "train"
    split_validation = "validation"
    split_test = "test"

    no_labels = 102
    dataset_name = "oxford_flowers102"
    dataset_version = "2.1.1"
    print("Dataset: {}".format(dataset_name))

    dataset_code = '{0:s}:{1:s}'.format(dataset_name, dataset_version)

    data_loaded = prepare_dataset(dataset_code=dataset_code,
                                  split_train=split_train,
                                  split_validation=split_validation,
                                  split_test=split_test,
                                  batch_size=batch_size,
                                  dataset_name=dataset_name,
                                  no_labels=no_labels)

    return data_loaded

###############################################################################
# DEEP WEEDS
###############################################################################
def load_deep_weeds_dataset(batch_size):

    train_prop = 70
    valid_prop = 10
#     test_prop=20

    split_train = "train[:{0:d}%]".format(train_prop)
    split_validation = "train[{0:d}:{1:d}%]".format(train_prop, train_prop + valid_prop)
    split_test = "train[{0:d}:{1:d}%]".format(train_prop + valid_prop, 100)

    no_labels = 9
    dataset_name = "deep_weeds"
    dataset_version = "3.0.0"
    print("Dataset: {}".format(dataset_name))

    dataset_code = '{0:s}:{1:s}'.format(dataset_name, dataset_version)

    data_loaded = prepare_dataset(dataset_code=dataset_code,
                                  split_train=split_train,
                                  split_validation=split_validation,
                                  split_test=split_test,
                                  batch_size=batch_size,
                                  dataset_name=dataset_name,
                                  no_labels=no_labels)

    return data_loaded

###############################################################################
# PLANT LEAVES
###############################################################################
def load_plant_leaves_dataset(batch_size):

    train_prop = 70
    valid_prop = 10
#     test_prop=20

    split_train = "train[:{0:d}%]".format(train_prop)
    split_validation = "train[{0:d}:{1:d}%]".format(train_prop, train_prop + valid_prop)
    split_test = "train[{0:d}:{1:d}%]".format(train_prop + valid_prop, 100)

    no_labels = 22
    dataset_name = "plant_leaves"
    dataset_version = "0.1.0"
    print("Dataset: {}".format(dataset_name))

    dataset_code = '{0:s}:{1:s}'.format(dataset_name, dataset_version)

    data_loaded = prepare_dataset(dataset_code=dataset_code,
                                  split_train=split_train,
                                  split_validation=split_validation,
                                  split_test=split_test,
                                  batch_size=batch_size,
                                  dataset_name=dataset_name,
                                  no_labels=no_labels)

    return data_loaded


###############################################################################
# CASSAVA
###############################################################################
def load_cassava_dataset(batch_size):

    split_train = "train"
    split_validation = "validation"
    split_test = "test"

    no_labels = 5
    dataset_name = "cassava"
    dataset_version = "0.1.0"
    print("Dataset: {}".format(dataset_name))

    dataset_code = '{0:s}:{1:s}'.format(dataset_name, dataset_version)

    data_loaded = prepare_dataset(dataset_code=dataset_code,
                                  split_train=split_train,
                                  split_validation=split_validation,
                                  split_test=split_test,
                                  batch_size=batch_size,
                                  dataset_name=dataset_name,
                                  no_labels=no_labels)

    return data_loaded


###############################################################################
# CATS VS DOGS
###############################################################################
def load_cats_vs_dogs_dataset(batch_size):

    train_prop = 70
    valid_prop = 10
#     test_prop=20

    split_train = "train[:{0:d}%]".format(train_prop)
    split_validation = "train[{0:d}:{1:d}%]".format(train_prop, train_prop + valid_prop)
    split_test = "train[{0:d}:{1:d}%]".format(train_prop + valid_prop, 100)

    no_labels = 2
    dataset_name = "cats_vs_dogs"
    dataset_version = "4.0.0"
    print("Dataset: {}".format(dataset_name))

    dataset_code = '{0:s}:{1:s}'.format(dataset_name, dataset_version)

    data_loaded = prepare_dataset(dataset_code=dataset_code,
                                  split_train=split_train,
                                  split_validation=split_validation,
                                  split_test=split_test,
                                  batch_size=batch_size,
                                  dataset_name=dataset_name,
                                  no_labels=no_labels)

    return data_loaded


###############################################################################
# BIRDS 2011
###############################################################################
def load_caltech_birds2011_dataset(batch_size):

    train_prop = 90
#     valid_prop = 10

    split_train = "train[:{0:d}%]".format(train_prop)
    split_validation = "train[{0:d}:{1:d}%]".format(train_prop, 100)
    split_test = "test"

    no_labels = 200
    dataset_name = "caltech_birds2011"
    dataset_version = "0.1.1"
    print("Dataset: {}".format(dataset_name))

    dataset_code = '{0:s}:{1:s}'.format(dataset_name, dataset_version)

    data_loaded = prepare_dataset(dataset_code=dataset_code,
                                  split_train=split_train,
                                  split_validation=split_validation,
                                  split_test=split_test,
                                  batch_size=batch_size,
                                  dataset_name=dataset_name,
                                  no_labels=no_labels)

    return data_loaded


###############################################################################
# CARS196
###############################################################################
def load_cars196_dataset(batch_size):

    train_prop = 90
#     valid_prop = 10

    split_train = "train[:{0:d}%]".format(train_prop)
    split_validation = "train[{0:d}:{1:d}%]".format(train_prop, 100)
    split_test = "test"

    no_labels = 196
    dataset_name = "cars196"
    dataset_version = "2.0.0"
    print("Dataset: {}".format(dataset_name))

    dataset_code = '{0:s}:{1:s}'.format(dataset_name, dataset_version)

    data_loaded = prepare_dataset(dataset_code=dataset_code,
                                  split_train=split_train,
                                  split_validation=split_validation,
                                  split_test=split_test,
                                  batch_size=batch_size,
                                  dataset_name=dataset_name,
                                  no_labels=no_labels)

    return data_loaded


###############################################################################
# MALARIA
###############################################################################
def load_malaria_dataset(batch_size):

    train_prop = 70
    valid_prop = 10
#     test_prop=20

    split_train = "train[:{0:d}%]".format(train_prop)
    split_validation = "train[{0:d}:{1:d}%]".format(train_prop, train_prop + valid_prop)
    split_test = "train[{0:d}:{1:d}%]".format(train_prop + valid_prop, 100)

    no_labels = 2
    dataset_name = "malaria"
    dataset_version = "1.0.0"
    print("Dataset: {}".format(dataset_name))

    dataset_code = '{0:s}:{1:s}'.format(dataset_name, dataset_version)
    data_loaded = prepare_dataset(dataset_code=dataset_code,
                                  split_train=split_train,
                                  split_validation=split_validation,
                                  split_test=split_test,
                                  batch_size=batch_size,
                                  dataset_name=dataset_name,
                                  no_labels=no_labels)

    return data_loaded


def prepare_dataset(dataset_code, split_train, split_validation, split_test,
                    batch_size, dataset_name, no_labels):

    data_train, info_train = tfds.load(dataset_code,
                                       as_supervised=True,
                                       split=split_train,
                                       shuffle_files=False,
                                       with_info=True,
                                       data_dir=TF_DATASETS_FOLDER)

    data_valid, info_valid = tfds.load(dataset_code,
                                       as_supervised=True,
                                       split=split_validation,
                                       shuffle_files=False,
                                       with_info=True,
                                       data_dir=TF_DATASETS_FOLDER)

    data_test, info_test = tfds.load(dataset_code,
                                     as_supervised=True,
                                     split=split_test,
                                     shuffle_files=False,
                                     with_info=True,
                                     data_dir=TF_DATASETS_FOLDER)

    if not os.path.exists(os.path.join(TF_DATASETS_FOLDER, dataset_name)):
        os.makedirs(os.path.join(TF_DATASETS_FOLDER, dataset_name))

    if not os.path.exists(os.path.join(INFO_DATASETS_FOLDER, dataset_name)):
        os.makedirs(os.path.join(INFO_DATASETS_FOLDER, dataset_name))

    info_train.write_to_directory(os.path.join(INFO_DATASETS_FOLDER, dataset_name))

    # Extracting and saving features
    train_size, y_train = extract_features_from_set(data_train, dataset_name, "train")
    valid_size, y_valid = extract_features_from_set(data_valid, dataset_name, "valid")
    test_size, y_test = extract_features_from_set(data_test, dataset_name, "test")

    #print("No. examples. TRAIN: {0:d}, VALID: {1:d}, TEST; {2:d}".format(train_size, valid_size, test_size))

    # Transforming output to one-hot encoding
    data_train = data_train.map(lambda image, label: (image, tf.one_hot(label, no_labels)))
    data_valid = data_valid.map(lambda image, label: (image, tf.one_hot(label, no_labels)))
    data_test = data_test.map(lambda image, label: (image, tf.one_hot(label, no_labels)))

    # RESIZING TO 224x224
    #print("Resizing data...")
    data_train = data_train.map(lambda image, label: transform_images(image, IMG_SIZE, label))
    data_valid = data_valid.map(lambda image, label: transform_images(image, IMG_SIZE, label))
    data_test = data_test.map(lambda image, label: transform_images(image, IMG_SIZE, label))

    img_shape = tf.compat.v1.data.get_output_shapes(data_train)[0]
    #print("Image shape: {}".format(img_shape))

    # Preaparing data for fitting
    data_train = data_train.batch(batch_size=batch_size).repeat()
    data_valid = data_valid.batch(batch_size=batch_size).repeat()
    data_test = data_test.batch(batch_size=batch_size).repeat()

    data_loaded = {
        "train": data_train,
        "validation": data_valid,
        "test": data_test,
        "train_size": train_size,
        "valid_size": valid_size,
        "test_size": test_size,
        "y_train": y_train,
        "y_valid": y_valid,
        "y_test": y_test,
        "img_shape": img_shape,
        "num_labels": no_labels

    }

    return data_loaded

def g(i):
    return extract_features(image=i[0], label=i[1])

def extract_features_from_set(data, dataset_name, set_name):
    """
    Invokes the extract_features method to extract the features of each image.
    This function is also used to calculate the real number of example by set,
    given that we are managing tf datasets and cannot retrieve directly the size.
    """
    
    from time import time
    features_file_path = os.path.join(FEATURES_DATASETS_FOLDER, '{0:s}_{1:s}_features.csv'.format(dataset_name, set_name))

    if os.path.exists(features_file_path):
        #print("Features file of {0:s} set already exists!".format(set_name))
        features_df = pd.read_csv(features_file_path)
        labels = features_df.label.values
        size = features_df.shape[0]
        return size, labels

    else:
        
        print("Extracting features from {0:s}".format(set_name), flush=True)
        start_time = time()
        feature_list, labels = list(), list()
        #for i, (image, label) in enumerate(tqdm(tfds.as_numpy(data))):
        #    feature_list.append(extract_features(image, i))
        #    labels.append(label)
            

        pool = Pool(processes=16)
        for feature in tqdm(pool.map(g, tfds.as_numpy(data)), total=len(data)):
            feature_list.append(feature)    
        
        duration = time() - start_time
        print("DURATION: " + dataset_name + " " + str(duration))
        #pool = multiprocessing.Pool()
        #features_list = pool.starmap(extract_features, tfds.as_numpy(data))
        
        labels = [label for (image, label) in tfds.as_numpy(data)]

        feature_set = pd.DataFrame(feature_list)
        feature_set['label'] = labels
        feature_set.to_csv(features_file_path)

        size = len(data)
        return size, labels