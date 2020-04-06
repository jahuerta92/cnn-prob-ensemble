# Nubes

- **1_train_CNNs.py and 1_train_CNNs.ipynb** -> Train VGG19, inceptionresnetv2, inceptionv3, DenseNet201 and Xception a number of times. Generates predictions and reports. We only considered the first three ones.

- **2_classification_experiments** -> Run experiments:
	-	**Experiment 1:** CNN (results extraction and plotting)
	-	**Experiment 2:** RF on estimators classification
	-	**Experiment 3:** Average of RF over estimators + CNN predictions
	-	**Experiment 4:** Standard classifiers on CNN predictions + RF over estimators
	-	**Experiment 5:** Standard classifiers on CNN predictions + RF over estimators + CEIL features