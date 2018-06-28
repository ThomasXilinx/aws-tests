#!/bin/bash

: "${AWS_ACCESS_KEY_ID:?Need to set AWS_ACCESS_KEY_ID}"
: "${AWS_SECRET_ACCESS_KEY:?Need to set AWS_SECRET_ACCESS_KEY}"
: "${AWS_DEFAULT_REGION:?Need to set AWS_DEFAULT_REGION}"
: "${AWS_DEFAULT_OUTPUT:?Need to set AWS_DEFAULT_OUTPUT}"
: "${1:?Need to provide an input xclbin file}"

file=$(basename ${1})
echo $file

if [ -f $file ]; then
   echo "File $file exists."
else
   echo "File $file does not exist."
fi

xclbinfile=$PWD/$file
awsxclbindir=$file.aws
mkdir $awsxclbindir
cd $awsxclbindir

$SDACCEL_DIR/tools/create_sdaccel_afi.sh -s3_bucket=myafibucket -s3_dcp_key=dcpfolder -s3_logs_key=logsfolder -xclbin=$xclbinfile

wait

~/aws-tests/afi_status.sh 

