#!/bin/bash
set -e

# flickr coco nuswide
cd code/
matlab -nojvm -nodesktop -r "demo_FOMH('16', 'flickr'); quit;"
cd ..

