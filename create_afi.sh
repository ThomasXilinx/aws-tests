#!/bin/bash

: "${1:?Need to provide an input xclbin file}"

xclbinfile=$(basename ${1})

if [ -f $xclbinfile ]; then
   echo "File $xclbinfile exists."
else
   echo "File $xclbinfile does not exist."
fi

extension="${xclbinfile##*.}"
filename="${xclbinfile%.*}"
awsxclbinfile=$filename

afibucket=xlnx-f1-developer/tbollaer
dcpfolder=xlnx-f1-developer/tbollaer/f1-dcp-folder
logsfolder=xlnx-f1-developer/tbollaer/f1-logs

$VITIS_DIR/tools/create_vitis_afi.sh -s3_bucket=$afibucket -s3_dcp_key=$dcpfolder -s3_logs_key=$logsfolder -xclbin=$xclbinfile -o=$awsxclbinfile



