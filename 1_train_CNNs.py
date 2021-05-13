# Training of base CNN classifiers

import os
import fire
import sys

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.applications.xception import Xception

from src.Model import make_prebuilt

from src.datasets import load_malaria_dataset
from src.datasets import load_cars196_dataset
from src.datasets import load_caltech_birds2011_dataset
from src.datasets import load_cats_vs_dogs_dataset
from src.datasets import load_cassava_dataset
from src.datasets import load_plant_leaves_dataset
from src.datasets import load_deep_weeds_dataset
from src.datasets import load_oxford_flowers102_dataset
from src.datasets import load_citrus_leaves_dataset
from src.datasets import load_plant_village_dataset


from src.Utils import train_model

import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

warnings.filterwarnings("ignore")

BATCH_SIZE = 64


def run_model(model_name, dataset_name):

    if dataset_name == "malaria":
        data_loaded = load_malaria_dataset(batch_size=BATCH_SIZE)

    elif dataset_name == "cars196":
        data_loaded = load_cars196_dataset(batch_size=BATCH_SIZE)

    elif dataset_name == "caltech_birds2011":
        data_loaded = load_caltech_birds2011_dataset(batch_size=BATCH_SIZE)

    elif dataset_name == "cats_vs_dogs":
        data_loaded = load_cats_vs_dogs_dataset(batch_size=BATCH_SIZE)

    elif dataset_name == "cassava":
        data_loaded = load_cassava_dataset(batch_size=BATCH_SIZE)

    elif dataset_name == "plant_leaves":
        data_loaded = load_plant_leaves_dataset(batch_size=BATCH_SIZE)

    elif dataset_name == "deep_weeds":
        data_loaded = load_deep_weeds_dataset(batch_size=BATCH_SIZE)

    elif dataset_name == "oxford_flowers102":
        data_loaded = load_oxford_flowers102_dataset(batch_size=BATCH_SIZE)

    elif dataset_name == "citrus_leaves":
        data_loaded = load_citrus_leaves_dataset(batch_size=BATCH_SIZE)

    elif dataset_name == "plant_village":
        data_loaded = load_plant_village_dataset(batch_size=BATCH_SIZE)

    else:
        print("dataset not found!")
        sys.exit(0)

    if model_name == "vgg19":
        # VGG19
        model_name = "vgg19"
        freeze_prop = 0.0
        model_builder = make_prebuilt(VGG19, freeze_prop, include_ceilometer=False)
        train_model(model_name, data_loaded, BATCH_SIZE, model_builder, freeze_prop, dataset_name)

    elif model_name == "inceptionresnetv2":

        # inceptionresnetv2
        model_name = "inceptionresnetv2"
        freeze_prop = .25
        model_builder = make_prebuilt(InceptionResNetV2, freeze_prop, include_ceilometer=False)
        train_model(model_name, data_loaded, BATCH_SIZE, model_builder, freeze_prop, dataset_name)

    elif model_name == "inceptionv3":

        # inceptionv3
        model_name = "inceptionv3"
        freeze_prop = .1
        model_builder = make_prebuilt(InceptionV3, freeze_prop, include_ceilometer=False)
        train_model(model_name, data_loaded, BATCH_SIZE, model_builder, freeze_prop, dataset_name)

    elif model_name == "densenet201":

        # densenet201
        model_name = "densenet201"
        freeze_prop = .25
        model_builder = make_prebuilt(DenseNet201, freeze_prop, include_ceilometer=False)
        train_model(model_name, data_loaded, BATCH_SIZE, model_builder, freeze_prop, dataset_name)

    elif model_name == "xceptionv1":

        # xceptionv1
        model_name = "xceptionv1"
        freeze_prop = .25
        model_builder = make_prebuilt(Xception, freeze_prop, include_ceilometer=False)
        train_model(model_name, data_loaded, BATCH_SIZE, model_builder, freeze_prop, dataset_name)


if __name__ == '__main__':
    fire.Fire(run_model)