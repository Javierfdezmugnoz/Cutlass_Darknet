#!/bin/bash

#from 0-31
BIT_FLIP=30
for idx_weight in `seq 302 1 1000`
do
    mkdir results/${idx_weight}_activationweight/
    ./bitflip yolov3.weights ${idx_weight} ${BIT_FLIP} yolov3_bitflip.weights
    # Execute YOLOv3 (10 000 iterations)
    ./darknet detector test ./cfg/coco.data ./cfg/yolov3.cfg ./yolov3_bitflip.weights -dont_show -ext_output <data/file_list.txt> results/results.csv
    cd results
    mv *.csv ${idx_weight}_activationweight/.
    cd ..
done
