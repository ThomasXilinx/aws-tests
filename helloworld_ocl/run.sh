#!/bin/bash

function run_on_f1 {
sudo sh -s -- <<EOF
        source /opt/xilinx/xrt/setup.sh
	./helloworld
EOF
}

make exe
run_on_f1

