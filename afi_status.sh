#!/bin/bash

AFI_ID=$(grep -Po '"'"FpgaImageId"'"\s*:\s*"\K([^"]*)' *_afi_id.txt)

aws ec2 describe-fpga-images --fpga-image-ids $AFI_ID

echo $AFI_ID
