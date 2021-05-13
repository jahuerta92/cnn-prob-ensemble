#!/bin/bash
for i in {1..5}
do
    echo "------------------------------Iteration: $i------------------------------"
    echo "Vgg19. $i run"
    python3 1_train_CNNs.py vgg19 oxford_flowers102
    echo "Inceptionresnetv2. $i run"
    python3 1_train_CNNs.py inceptionresnetv2 oxford_flowers102
    echo "inceptionv3. $i run"
    python3 1_train_CNNs.py inceptionv3 oxford_flowers102
    echo "densenet201. $i run"
    python3 1_train_CNNs.py densenet201 oxford_flowers102
    echo "xceptionv1. $i run"
    python3 1_train_CNNs.py xceptionv1 oxford_flowers102
done


