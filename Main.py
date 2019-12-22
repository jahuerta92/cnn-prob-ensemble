from Utils import load_data, save_results, fit_model
from Model import *
from keras.applications.vgg19 import VGG19

data_dir = './data'

data = load_data(data_dir)

model_dir = "./results"
model_name = 'vgg19'

# model, dgen = fit_model(data['train'], data['valid'], make_prebuilt(VGG19),
#         model_name=model_name, model_dir=model_name)

model, dgen = fit_model(data['train'], data['valid'], make_dummy,
                        model_name=model_name, model_dir=model_name, max_epochs=2)

save_results(model_dir, model_name, encoder=data['label_encoder'],
             normalizer=dgen, test_data=data['test'])
