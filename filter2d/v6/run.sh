#!/bin/bash

function run_on_f1 {
sudo sh -s -- <<EOF
        source /opt/xilinx/xrt/setup.sh
	./Filter2D.exe -x xclbin/fpga.3k.hw.awsxclbin -i img/picadilly_1080p.bmp -n 5
EOF
}

make
run_on_f1

