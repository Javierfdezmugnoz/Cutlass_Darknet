#!/bin/bash

ROOT_PATH=${PWD}
mkdir summary
for i_loop in `seq 0 1 1`
do
    cp ${ROOT_PATH}/cmp_file ${ROOT_PATH}/${i_loop}_activationweight/.
    cd ${ROOT_PATH}/${i_loop}_activationweight
    ./cmp_file
    cp max_undetected_cmp.csv ${ROOT_PATH}/summary/${i_loop}_undetected_cmp.csv
    cd $ROOT_PATH
done
cd summary
cat *csv > 0_summarized.csv
