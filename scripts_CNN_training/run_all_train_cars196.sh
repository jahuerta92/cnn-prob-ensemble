#!/bin/bash
for i in {1..5}
do
    echo "------------------------------Iteration: $i------------------------------"
    echo "Vgg19. $i run"
    python3 1_train_CNNs.py vgg19 cars196
    echo "Inceptionresnetv2. $i run"
    python3 1_train_CNNs.py inceptionresnetv2 cars196
    echo "inceptionv3. $i run"
    python3 1_train_CNNs.py inceptionv3 cars196
    echo "densenet201. $i run"
    python3 1_train_CNNs.py densenet201 cars196
    echo "xceptionv1. $i run"
    python3 1_train_CNNs.py xceptionv1 cars196
done


