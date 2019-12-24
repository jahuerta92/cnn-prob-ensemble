from Utils import *
from Model import *

data_dir = './data'

data = load_data(data_dir)

model_dir = "./results"

img_train, ceil_train, y_train = data['train']
data_generator = ImageDataGenerator(featurewise_center=True, samplewise_center=True)
data_generator.fit(img_train)

model_name = 'rcropnetv1'

fit_model(data['train'], data['valid'], data_generator,
          make_rcropnetv1, model_name=model_name,
          model_dir=model_dir, n_outputs=6)
save_results(model_dir, model_name, data['label_encoder'],
             data_generator, test_data=data['test'], n_outputs=6)
# model, dgen = fit_model(data['train'], data['valid'], make_prebuilt(VGG19),
#         model_name=model_name, model_dir=model_name)

#model, dgen = fit_model(data['train'], data['valid'], make_dummy,
#                        model_name=model_name, model_dir=model_name, max_epochs=2)

#save_results(model_dir, model_name, encoder=data['label_encoder'],
#             normalizer=dgen, test_data=data['test'])
