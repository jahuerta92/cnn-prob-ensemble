import tensorflow as tf
config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)    
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.90
sess = tf.Session(config = config)

import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3' 
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.applications.xception import Xception
from Model import *
from Utils import *
from imblearn.over_sampling import *


NUM_EXECUTIONS = 10
BATCH_SIZE = 32

INCLUDE_CEILOMETER = False

experiment_ceil_text = "ceil"
if not INCLUDE_CEILOMETER:
    experiment_ceil_text = "no_ceil"


NUM_EXECUTIONS = 5

file_dir = "data"
model_dir = "results"

# TODO 
data = load_data(file_dir, oversampler=RandomOverSampler())

data_generator = make_data_generator(data['train'])

for n_execution in range(NUM_EXECUTIONS):
    
    print("EXECUTION: %d" % n_execution)
    print("INCLUDE CEIL FEATURES: %r" % INCLUDE_CEILOMETER)

    # https://github.com/keras-team/keras/issues/2131
    model_name = 'vgg19_%s' % experiment_ceil_text
    model, model_file_name = fit_model(data['train'], data['valid'], data_generator, 
                                       make_prebuilt(VGG19,.1, include_ceilometer=False), model_name=model_name, 
                                       model_dir=model_dir, batch_size=BATCH_SIZE, include_ceilometer=INCLUDE_CEILOMETER)

    save_results(model_dir, model_file_name, data, data_generator, include_ceilometer=INCLUDE_CEILOMETER)
    
    model_name = 'inceptionresnetv2_%s' % experiment_ceil_text
    model, model_file_name = fit_model(data['train'], data['valid'], data_generator,
                            make_prebuilt(InceptionResNetV2,.25, include_ceilometer=False), model_name=model_name,
                            model_dir=model_dir, batch_size=BATCH_SIZE, include_ceilometer=INCLUDE_CEILOMETER)
    save_results(model_dir, model_file_name, data, data_generator, include_ceilometer=INCLUDE_CEILOMETER)

    model_name = 'inceptionv3_%s' % experiment_ceil_text
    model, model_file_name = fit_model(data['train'], data['valid'], data_generator,
                            make_prebuilt(InceptionV3,.1, include_ceilometer=False), model_name=model_name,
                            model_dir=model_dir, batch_size=BATCH_SIZE, include_ceilometer=INCLUDE_CEILOMETER)

    save_results(model_dir, model_file_name, data, data_generator, include_ceilometer=INCLUDE_CEILOMETER)

    
    model_name = 'densenet201_%s' % experiment_ceil_text
    model, model_file_name = fit_model(data['train'], data['valid'], data_generator,
                            make_prebuilt(DenseNet201,.25, include_ceilometer=False), model_name=model_name,
                            model_dir=model_dir, batch_size=BATCH_SIZE, include_ceilometer=INCLUDE_CEILOMETER)

    save_results(model_dir, model_file_name, data, data_generator, include_ceilometer=INCLUDE_CEILOMETER)

    model_name = 'xceptionv1_%s' % experiment_ceil_text
    model, model_file_name = fit_model(data['train'], data['valid'], data_generator,
                            make_prebuilt(Xception,.25, include_ceilometer=False), model_name=model_name,
                            model_dir=model_dir, batch_size=BATCH_SIZE, include_ceilometer=INCLUDE_CEILOMETER)

    save_results(model_dir, model_file_name, data, data_generator, include_ceilometer=INCLUDE_CEILOMETER)

