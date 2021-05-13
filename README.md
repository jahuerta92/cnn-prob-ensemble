# Fusion of CNNs and statistical indicators to improve image classification

### Abstract

Convolutional Networks have dominated the field of computer vision for the last ten years, exhibiting extremely powerful feature extraction capabilities and outstanding classification performance. The main strategy to prolong this trend relies on further upscaling networks in size. However, costs increase rapidly while performance improvements may be marginal. We hypothesise that adding additional sources of information can help to increase performance and that this approach may be more cost-effective than building bigger networks, which involve higher training time, larger parametrization space and high computational resources needs. In this paper, an ensemble method is proposed for accurate image classification, fusing automatically detected features through a Convolutional Neural Network and a set of manually defined statistical indicators. Through a combination of the predictions of a CNN and a secondary classifier trained on statistical features, better classification performance can be cheaply achieved. We test five different CNN architectures and multiple learning algorithms on a diverse number of datasets to validate our proposal. According to the results, the inclusion of additional indicators and an ensemble classification approach helps to increase the performance all datasets.

- **1_train_CNNs.py <model_name> <dataset_name>:** This file is used to train one specific architecture in a given domain. Folder **scripts_CNN_training** contains the necessary scripts to run 5 times all architectures for all datasets.
- **2_classification_experiments_all_cnns.py <dataset_name>:**. It is used to run all the experiments once the CNNs have been trained. The statistical features are extracted for all datasets and different classification algorithms are trained with these feature. Then, different classifiers and the average are tested to build the fusion approach, combining the best classifier trained on the statistical features and the CNN output. 
- **3_classification_experiments_results_collection.ipynb:** This file is used to extract and represent all the results obtained.
- **4_ablation_study_stat_out.py <dataset_name> <cnn_model>:**  This file runs the ablation study for each statistical feature, each dataset and each CNN architecture.


# Publication

This research has been submitted to the Information Fusion journal. A preliminary version can be found at:

> @article{huertas2020fusion,
  title={Fusion of CNNs and statistical indicators to improve image classification},
  author={Huertas-Tato, Javier and Mart{\'\i}n, Alejandro and Fierrez, Julian and Camacho, David},
  journal={arXiv preprint arXiv:2012.11049},
  year={2020}
}

