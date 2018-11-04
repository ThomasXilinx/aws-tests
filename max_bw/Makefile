
KERNEL  	 = CopyKernel
NKERNEL          = 1

DEVICE          ?= f1-dynamic
TARGET          ?= hw_emu
NKERNEL         ?= 1

APP_SOURCE_FILES = ./src/cmdlineparser.cpp ./src/logger.cpp ./src/xcl2.cpp ./src/host.cpp
APP_HEADER_FILES = ./src/cmdlineparser.h   ./src/logger.h   ./src/xcl2.hpp 
KNL_SOURCE_FILES = ./src/kernel.cpp  
KNL_HEADER_FILES =  

ifeq ($(TARGET),sw_emu)
XOCC_COMP_OPTIONS := -O3
else
XOCC_COMP_OPTIONS := -g --profile_kernel data:all:all:all --xp param:compiler.hlsDataflowStrictMode=warning
endif

XOCC_LINK_OPTIONS := --sp CopyKernel_1.m_axi_gmem0:bank0 --sp CopyKernel_1.m_axi_gmem1:bank1 		

APP		= app.exe

# -----------------------------------------------------------------------------

XO_FILE			:= xclbin/${KERNEL}.${TARGET}.${DEVICE}.xo
XCLBIN_FILE 	:= xclbin/fpga.${NKERNEL}k.${TARGET}.${DEVICE}.xclbin

ifeq ($(DEVICE),f1-dynamic)
PLATFORM := /home/tbollaer/AWS/aws-fpga-v138/SDAccel/aws_platform/xilinx_aws-vu9p-f1_dynamic_5_0/xilinx_aws-vu9p-f1_dynamic_5_0.xpfm
endif
ifeq ($(DEVICE),vcu1525)
PLATFORM := xilinx_vcu1525_dynamic_5_1
endif

# -----------------------------------------------------------------------------

# Compile and build the host application (.exe file)
${APP}: ${APP_SOURCE_FILES} ${APP_HEADER_FILES}
	xcpp -O2 -fopenmp -o $@ ${APP_SOURCE_FILES} \
	-I${XILINX_SDX}/include -I${XILINX_SDX}/runtime/include/1_2 -L${XILINX_SDX}/runtime/lib/x86_64 -L${XILINX_SDX}/lib/lnx64.o -lOpenCL -pthread

# Compile the kernel (.xo file)
${XO_FILE}: ${KNL_SOURCE_FILES} ${KNL_HEADER_FILES}
	mkdir -p xclbin
	xocc -c -t ${TARGET} ${XOCC_COMP_OPTIONS} -s --platform ${PLATFORM} -k ${KERNEL} ${KNL_SOURCE_FILES} -o $@
	

# Link the FPGA binary (.xclbin file)
${XCLBIN_FILE}: ${XO_FILE}
	xocc -l -t ${TARGET} ${XOCC_COMP_OPTIONS} ${XOCC_LINK_OPTIONS} --nk ${KERNEL}:${NKERNEL} -s --platform ${PLATFORM} $<  -o $@ \

# Build the host application and XCLBIN
app: ${APP}
xo: ${XO_FILE}
xclbin: ${XCLBIN_FILE}
build: ${XCLBIN_FILE} ${APP} 

# Execute the accelerated application
check: ${XCLBIN_FILE} emconfig.json ${APP} 
	XCL_EMULATION_MODE=${TARGET} ./${APP} -x ${XCLBIN_FILE} -l 256 -i 4 -n 1024

# Create the emulation config file
emconfig.json:
	emconfigutil --platform ${PLATFORM} --nd 1

# Run on the board farm
sdxbf: ${APP} 
	rm -rf sdxbf* sdaccel_*
	cp xclbin/fpga.${NKERNEL}k.hw.vcu1525.xclbin .
	/proj/dsamgr/prod/bin/sdxbf submit-test \
		-w . \
		-p xilinx:vcu1525:dynamic:5.1 \
		-a ' -x fpga.${NKERNEL}k.hw.vcu1525.xclbin -l 128 -i 4000 -n  262144' \
		-t xsj \

	cp sdxbf_results/sdxbf_test/TEST_WORK_lnx64/*.csv .
	make profile
	tail sdxbf_results/sdxbf_test/TEST_WORK_lnx64/output.log 	

#		-a ' -x fpga.${NKERNEL}k.hw.vcu1525.xclbin -l 128 -i  500 -n 2097152' 
#		-a ' -x fpga.${NKERNEL}k.hw.vcu1525.xclbin -l 128 -i 1000 -n 1048576' 
#		-a ' -x fpga.${NKERNEL}k.hw.vcu1525.xclbin -l 128 -i 2000 -n  524288' 
#		-a ' -x fpga.${NKERNEL}k.hw.vcu1525.xclbin -l 128 -i 4000 -n  262144' 

# -------------------------------------------------


sw_emu:
	make check TARGET=sw_emu 

hw_emu:
	make check TARGET=hw_emu

hw:
	make build TARGET=hw

# -------------------------------------------------

profile:
	# sda2protobuf sdaccel_profile_summary.csv
	# sda2wdb sdaccel_timeline_trace.csv	
	sdx_analyze profile sdaccel_profile_summary.csv
	sdx_analyze profile sdaccel_profile_summary.csv -f html	
	sdx_analyze trace sdaccel_timeline_trace.csv

# -------------------------------------------------

clean:
	rm -rf ${APP} sdaccel_* emconfig.json emulation* xsim* *.wcfg *.wdb .Xil prj sdx_* *.log xclbin _xocc*
	
