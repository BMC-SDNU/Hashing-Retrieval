#!/bin/bash

cd code/ &&
matlab -nojvm -nodesktop -r "demo_STMH('16', '16', 'flickr'); quit;" 
cd ..