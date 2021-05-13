#!/bin/bash
for i in {1..5}
do
    echo "------------------------------Iteration: $i------------------------------"
    echo "Vgg19. $i run"
    python3 1_train_CNNs.py vgg19 caltech_birds2011
    echo "Inceptionresnetv2. $i run"
    python3 1_train_CNNs.py inceptionresnetv2 caltech_birds2011
    echo "inceptionv3. $i run"
    python3 1_train_CNNs.py inceptionv3 caltech_birds2011
    echo "densenet201. $i run"
    python3 1_train_CNNs.py densenet201 caltech_birds2011
    echo "xceptionv1. $i run"
    python3 1_train_CNNs.py xceptionv1 caltech_birds2011
done


