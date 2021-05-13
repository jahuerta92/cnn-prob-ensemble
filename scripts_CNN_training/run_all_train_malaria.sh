#!/bin/bash
for i in {1..5}
do
    echo "------------------------------Iteration: $i------------------------------"
    echo "Vgg19. $i run"
    python3 1_train_CNNs.py vgg19 malaria
    echo "Inceptionresnetv2. $i run"
    python3 1_train_CNNs.py inceptionresnetv2 malaria
    echo "inceptionv3. $i run"
    python3 1_train_CNNs.py inceptionv3 malaria
    echo "densenet201. $i run"
    python3 1_train_CNNs.py densenet201 malaria
    echo "xceptionv1. $i run"
    python3 1_train_CNNs.py xceptionv1 malaria
done


