#!/bin/bash
set -e

# flickr coco nuswide
cd code/
matlab -nojvm -nodesktop -r "demo_EPAMH('16', 'flickr'); quit;"
cd ..

