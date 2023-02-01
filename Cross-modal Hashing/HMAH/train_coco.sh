#!/bin/bash
set -e

for i in 16 32 64 128 256
do
  echo "Start Ours algorithm"
  CUDA_VISIBLE_DEVICES=0 python train.py --nbits $i --dataset COCO --n_class 80 --txt_dim 2000 --tea_epochs 100 --stu_epochs 100
  echo "End Ours algorithm"
  cd matlab
  matlab -nojvm -nodesktop -r "curve($i, 'COCO'); quit;"
  cd ..
done

