from Utils import *
from Model import *
import tensorflow as tf

config = tf.compat.v1.ConfigProto(log_device_placement=False, allow_soft_placement=True)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.90
sess = tf.compat.v1.Session(config=config)

data_dir = './data'

data = load_data(data_dir)

model_dir = "./results"

img_train, ceil_train, y_train = data['train']
label_num = y_train.shape[1]
ceil_features = ceil_train.shape[1]
img_shape = img_train.shape[1:]

model = make_rcropnetv1(64)(img_shape, ceil_features, label_num)
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.0001, clipnorm=1.),
              metrics=['accuracy'])
print('Layers: %d' % len(model.layers))
model.summary()
# model, dgen = fit_model(data['train'], data['valid'], make_prebuilt(VGG19),
#         model_name=model_name, model_dir=model_name)

#model, dgen = fit_model(data['train'], data['valid'], make_dummy,
#                        model_name=model_name, model_dir=model_name, max_epochs=2)

#save_results(model_dir, model_name, encoder=data['label_encoder'],
#             normalizer=dgen, test_data=data['test'])
