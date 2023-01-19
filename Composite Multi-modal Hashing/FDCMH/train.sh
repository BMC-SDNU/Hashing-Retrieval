#!/bin/bash
set -e

# flickr coco nuswide
cd code/
matlab -nojvm -nodesktop -r "demo_FDCMH('16', 'flickr'); quit;"
cd ..

