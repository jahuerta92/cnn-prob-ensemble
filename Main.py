from Utils import load_data, save_results
from Model import *
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from random import seed

data_dir = './data'

data = load_data(data_dir)

img_train, ceil_train, y_train = data['train']
img_valid, ceil_valid, y_valid = data['valid']

data_format = 'channels_last'
label_num = y_train.shape[1]
ceil_features = ceil_train.shape[1]
img_shape = img_train.shape[1:]

#base = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(height, width, channels))
#ceil_input = Input(shape= (ceil_features,))

#x = Dense(16, activation="relu")(ceil_input)
#x = Dropout(0.5)(x)

#y = GlobalAveragePooling2D()(base.output)
#x = Concatenate()([x, y])
#x = Dense(2048)(x)
#x = Dropout(0.5)(x)
#x = BatchNormalization()(x)
#x = Activation('relu')(x)

#predictions = Dense(label_num, activation="softmax")(x)

#disabled = floor(len(base.layers)*0.5)
#for layer in base.layers[:disabled]:
#    if 'batch_normalization' not in layer.name:
#        layer.trainable = False

model = make_vgg19(img_shape, ceil_features, label_num) #Model([base.input, ceil_input], predictions)

# Compilar el modelo
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=1e-5, clipnorm=1.),
              metrics=['accuracy'])
print('Layers: %d' % len(model.layers))
model.summary()

dgen = ImageDataGenerator(featurewise_center=True, samplewise_center=True,
                          rotation_range=180, width_shift_range=.2,
                          height_shift_range=.2, brightness_range=[.5,1.0],
                          zoom_range=[.7, 1.0], shear_range=45,
                          fill_mode='nearest', horizontal_flip=True,
                          vertical_flip=True)
dgen.fit(img_train)

# Entrenar el modelo
seed(1)
model_name = "inceptionresnetv2"
model_dir = './results'
model.fit_generator(dgen.flow((img_train, ceil_train), y_train, batch_size=64),
                    steps_per_epoch=len(img_train) / 64,
                    epochs=200,
                    verbose=2,
                    validation_data=dgen.flow((img_valid, ceil_valid), y_valid),
                    callbacks=[ModelCheckpoint('%s/%s' % (model_dir, '%s_model.h5' % model_name),
                                               monitor='val_acc', save_best_only=True)])

save_results(model_dir, model_name, encoder=data['label_encoder'],
             normalizer=dgen, test_data=data['test'])
