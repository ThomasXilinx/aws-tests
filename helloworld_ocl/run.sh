#!/bin/bash

make exe
sudo sh
source /opt/Xilinx/SDx/2017.4.rte.dyn/setup.sh
fpga-load-local-image -S 0 -I agfi-0724dbce53d0f1ba6 # load another AFI (filter2d)
./helloworld
exit
