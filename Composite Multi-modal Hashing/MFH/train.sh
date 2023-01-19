#!/bin/bash
set -e

# flickr coco nuswide
cd code/
matlab -nojvm -nodesktop -r "demo_MFH('16', 'flickr'); quit;"
cd ..

