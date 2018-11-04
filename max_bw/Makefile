
APP          = app.exe
KERNEL       = CopyKernel
NKERNEL      = 1
TARGET      ?= hw_emu
PLATFORM    ?= ${AWS_PLATFORM}
#PLATFORM   ?= xilinx_aws-vu9p-f1-04261818_dynamic_5_0
#PLATFORM   ?= xilinx_u200_xdma_201830_1
#PLATFORM   ?= xilinx_vcu1525_dynamic_5_1

# -----------------------------------------------------------------------------

APP_SOURCE_FILES = ./src/cmdlineparser.cpp ./src/logger.cpp ./src/xcl2.cpp ./src/host.cpp
APP_HEADER_FILES = ./src/cmdlineparser.h   ./src/logger.h   ./src/xcl2.hpp 
KNL_SOURCE_FILES = ./src/kernel.cpp  
KNL_HEADER_FILES =  

# -----------------------------------------------------------------------------

XOCC_COMP_OPTIONS += -g --profile_kernel data:all:all:all --profile_kernel stall:all:all:all 
XOCC_LINK_OPTIONS += --sp CopyKernel_1.m_axi_port0:bank0 --sp CopyKernel_1.m_axi_port1:bank1 		

APP_COMP_OPTIONS = -I${XILINX_SDX}/include -std=c++14 
APP_LINK_OPTIONS = -I${XILINX_SDX}/runtime/include/1_2 -L${XILINX_SDX}/runtime/lib/x86_64 -L${XILINX_SDX}/lib/lnx64.o -lxilinxopencl -pthread
#APP_LINK_OPTIONS = -I${XILINX_XRT}/include/ -L${XILINX_XRT}/lib/ -lxilinxopencl -lpthread -lrt -lstdc++

# -----------------------------------------------------------------------------

XO_FILE = xclbin/${KERNEL}.${TARGET}.xo
XCLBIN_FILE = xclbin/fpga.${NKERNEL}k.${TARGET}.xclbin
TMP_DIR = builddir.${NKERNEL}k.${TARGET}

# -----------------------------------------------------------------------------

# Compile and build the host application (.exe file)
${APP}: ${APP_SOURCE_FILES} ${APP_HEADER_FILES}
	xcpp -O2 -o $@ ${APP_SOURCE_FILES} ${APP_COMP_OPTIONS} ${APP_LINK_OPTIONS}

# Compile the kernel (.xo file)
${XO_FILE}: ${KNL_SOURCE_FILES} ${KNL_HEADER_FILES}
	mkdir -p xclbin
	xocc -c -t ${TARGET} ${XOCC_COMP_OPTIONS} -s --temp_dir ${TMP_DIR} --platform ${PLATFORM} -k ${KERNEL} ${KNL_SOURCE_FILES} -o $@
	
# Link the FPGA binary (.xclbin file)
${XCLBIN_FILE}: ${XO_FILE}
	xocc -l -t ${TARGET} ${XOCC_LINK_OPTIONS} -s --temp_dir ${TMP_DIR} --platform ${PLATFORM} --nk ${KERNEL}:${NKERNEL} $< -o $@ 

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
	sdx_analyze trace sdaccel_timeline_trace.csv
	sdx_analyze profile sdaccel_profile_summary.csv
	sdx_analyze profile sdaccel_profile_summary.csv -f html
	#sdx -workspace w -report sdaccel_timeline_trace.wdb &

# -------------------------------------------------

clean:
	rm -rf ${APP} sdaccel_* emconfig.json emulation* xsim* *.wcfg *.wdb .Xil prj sdx_* _x .run *.log xclbin _xocc*
	
