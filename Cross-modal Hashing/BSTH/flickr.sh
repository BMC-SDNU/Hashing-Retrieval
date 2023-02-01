#!/bin/bash
set -e
METHOD='BSTH'
bits=(16 32 64 128)

for i in ${bits[*]}; do
  echo "**********Start ${METHOD} algorithm**********"
  CUDA_VISIBLE_DEVICES=0 python train_transformer.py --nbit $i \
                                                     --dataset flickr \
                                                     --epochs 140 \
                                                     --dropout 0.5 \
                                                     --nhead 1 \
                                                     --num_layer 2 \

  echo "**********End ${METHOD} algorithm**********"

  echo "**********Start ${METHOD} evaluate**********"
  cd matlab
  matlab -nojvm -nodesktop -r "curve($i, 'flickr'); quit;"
  cd ..
  echo "**********End ${METHOD} evaluate**********"
done
