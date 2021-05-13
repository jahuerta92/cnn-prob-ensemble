#!/bin/bash
for i in {1..5}
do
    echo "------------------------------Iteration: $i------------------------------"
    echo "Vgg19. $i run"
    python3 1_train_CNNs.py vgg19 plant_village
    echo "Inceptionresnetv2. $i run"
    python3 1_train_CNNs.py inceptionresnetv2 plant_village
    echo "inceptionv3. $i run"
    python3 1_train_CNNs.py inceptionv3 plant_village
    echo "densenet201. $i run"
    python3 1_train_CNNs.py densenet201 plant_village
    echo "xceptionv1. $i run"
    python3 1_train_CNNs.py xceptionv1 plant_village
done


