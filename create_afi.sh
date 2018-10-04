#!/bin/bash

#: "${AWS_ACCESS_KEY_ID:?Need to set AWS_ACCESS_KEY_ID}"
#: "${AWS_SECRET_ACCESS_KEY:?Need to set AWS_SECRET_ACCESS_KEY}"
#: "${AWS_DEFAULT_REGION:?Need to set AWS_DEFAULT_REGION}"
#: "${AWS_DEFAULT_OUTPUT:?Need to set AWS_DEFAULT_OUTPUT}"
#: "${1:?Need to provide an input xclbin file}"

file=$(basename ${1})
echo $file

if [ -f $file ]; then
   echo "File $file exists."
else
   echo "File $file does not exist."
fi

xclbinfile=$PWD/$file
awsxclbindir=afi.$file
mkdir $awsxclbindir
cd $awsxclbindir

$SDACCEL_DIR/tools/create_sdaccel_afi.sh -s3_bucket=aws-xlnx-f1-developer -s3_dcp_key=pavan@xilinx.com/f1-dcp-folder -s3_logs_key=pavan@xilinx.com/logs -xclbin=$xclbinfile

wait

mkdir ../afi
mkdir ../afi/info
cp *.awsxclbin ../afi
cp *.txt ../afi/info

echo "Use ~/aws-tests/afi_status.sh to check status of the AFI"


