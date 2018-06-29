#!/bin/bash

make
sudo sh
source /opt/Xilinx/SDx/2017.4.rte.dyn/setup.sh
./Filter2D.exe -x xclbin/fpga.3k.hw.awsxclbin -i img/picadilly_1080p.bmp -n 5
exit
