 ../../build/tools/caffe train -solver model_simple/solver.prototxt -weights model_simple/flownet_official.caffemodel -gpu 1 2>&1 | tee -a  logfile
