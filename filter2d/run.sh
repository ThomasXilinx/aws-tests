#!/bin/bash

make
sudo sh
source /opt/Xilinx/SDx/2017.4.rte.dyn/setup.sh
fpga-load-local-image -S 0 -I agfi-019bdc344beb71ad4 # load another AFI (helloworld)
./Filter2D.exe -x xclbin/fpga.3k.hw.awsxclbin -i img/picadilly_1080p.bmp -n 5
exit
