#!/bin/bash

make exe
sudo sh
source /opt/Xilinx/SDx/2017.4.rte.dyn/setup.sh
./helloworld
exit
