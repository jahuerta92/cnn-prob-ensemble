#!/bin/bash
for i in {1..5}
do
    echo "------------------------------Iteration: $i------------------------------"
    echo "Vgg19. $i run"
    python3 1_train_CNNs.py vgg19 cats_vs_dogs
    echo "Inceptionresnetv2. $i run"
    python3 1_train_CNNs.py inceptionresnetv2 cats_vs_dogs
    echo "inceptionv3. $i run"
    python3 1_train_CNNs.py inceptionv3 cats_vs_dogs
    echo "densenet201. $i run"
    python3 1_train_CNNs.py densenet201 cats_vs_dogs
    echo "xceptionv1. $i run"
    python3 1_train_CNNs.py xceptionv1 cats_vs_dogs
done


