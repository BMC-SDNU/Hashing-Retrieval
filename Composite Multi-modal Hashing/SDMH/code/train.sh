#!/bin/bash
set -e


for i in flickr nuswide coco
do
    for j in 104545 109000 100090 303535 307000 300070 502525 505000 500050
    do
        for k in kmeans topk dual
        do
            matlab -nojvm -nodesktop -r "MMH_SDMH_twt('$i', '$j', '$k', '$USER'); quit;"
        done
    done
done
