#!/bin/bash
for i in {1..5}
do
    echo "------------------------------Iteration: $i------------------------------"
    echo "Vgg19. $i run"
    python3 1_train_CNNs.py vgg19 plant_leaves
    echo "Inceptionresnetv2. $i run"
    python3 1_train_CNNs.py inceptionresnetv2 plant_leaves
    echo "inceptionv3. $i run"
    python3 1_train_CNNs.py inceptionv3 plant_leaves
    echo "densenet201. $i run"
    python3 1_train_CNNs.py densenet201 plant_leaves
    echo "xceptionv1. $i run"
    python3 1_train_CNNs.py xceptionv1 plant_leaves
done


