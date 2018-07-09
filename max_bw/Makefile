
KERNEL  	    = CopyKernel
NKERNEL 	    = 1
DEVICE  	    = xilinx_aws-vu9p-f1_4ddr-xpr-2pr_4_0
PLATFORM 	    = ${AWS_PLATFORM}

APP_SOURCE_FILES = ./src/cmdlineparser.cpp ./src/logger.cpp ./src/xcl2.cpp ./src/host.cpp
APP_HEADER_FILES = ./src/cmdlineparser.h ./src/logger.h ./src/xcl2.hpp 
KNL_SOURCE_FILES = ./src/kernel.cpp  
KNL_HEADER_FILES =  

sw_emu_XOCC_OPTIONS = -O3 --xp "param:compiler.preserveHlsOutput=1" --xp "param:compiler.generateExtraRunData=true"
hw_emu_XOCC_OPTIONS = --debug --xp "param:compiler.preserveHlsOutput=1" --xp "param:compiler.generateExtraRunData=true"

XOCC_LINK_OPTIONS   = \
        --xp misc:map_connect=add.kernel.CopyKernel_1.M_AXI_GMEM0.core.OCL_REGION_0.M00_AXI \
        --xp misc:map_connect=add.kernel.CopyKernel_1.M_AXI_GMEM1.core.OCL_REGION_0.M01_AXI \


APP 				= app.exe

# -----------------------------------------------------------------------------

XO_FILE	= xclbin/${KERNEL}.${TARGET}.${DEVICE}.xo
XCLBIN_FILE = xclbin/fpga.${NKERNEL}k.${TARGET}.${DEVICE}.xclbin

# -----------------------------------------------------------------------------

# Compile and build the host application (.exe file)
${APP}: ${APP_SOURCE_FILES} ${APP_HEADER_FILES}
	xcpp -O2 -fopenmp -o $@ ${APP_SOURCE_FILES} \
	-I${XILINX_SDX}/include -I${XILINX_SDX}/runtime/include/1_2 -L${XILINX_SDX}/runtime/lib/x86_64 -L${XILINX_SDX}/lib/lnx64.o -lOpenCL -pthread

# Compile the kernel (.xo file)
${XO_FILE}: ${KNL_SOURCE_FILES} ${KNL_HEADER_FILES}
	mkdir -p xclbin
	xocc -c -t ${TARGET} ${${TARGET}_XOCC_OPTIONS} -s --platform ${PLATFORM} -k ${KERNEL} ${KNL_SOURCE_FILES} -o $@
	

# Link the FPGA binary (.xclbin file)
${XCLBIN_FILE}: ${XO_FILE}
	xocc -l -t ${TARGET} ${${TARGET}_XOCC_OPTIONS} ${XOCC_LINK_OPTIONS} --nk ${KERNEL}:${NKERNEL} -s --platform ${PLATFORM} $<  -o $@ \

# Build both the host application and XCLBIN
build: ${XCLBIN_FILE} ${APP} 

# Execute the accelerated application
check: ${XCLBIN_FILE} emconfig.json ${APP} 
	XCL_EMULATION_MODE=${TARGET} ./${APP} -x ${XCLBIN_FILE} -l 256 -i 4 -n 1024

# Create the emulation config file
emconfig.json:
	emconfigutil --platform ${PLATFORM} --nd 1

# -------------------------------------------------


sw_emu:
	make check TARGET=sw_emu 

hw_emu:
	make check TARGET=hw_emu

hw:
	make build TARGET=hw

# -------------------------------------------------

profile:
	sda2protobuf sdaccel_profile_summary.csv
	sda2wdb sdaccel_timeline_trace.csv	

# -------------------------------------------------

clean:
	rm -rf ${APP} sdaccel_* emconfig.json emulation* xsim* *.wcfg *.wdb .Xil prj sdx_* *.log xclbin _xocc*
	
