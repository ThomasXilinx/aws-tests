#!/bin/bash

cd $HOME

git clone https://github.com/aws/aws-fpga.git $AWS_FPGA_REPO_DIR
cd $AWS_FPGA_REPO_DIR
source sdaccel_setup.sh

export PLATFORM_REPO_PATHS=$AWS_FPGA_REPO_DIR/SDAccel/aws_platform

cd $HOME
