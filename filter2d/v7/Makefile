
KERNEL  	    = Filter2DKernel
NKERNEL 	    = 3
#PLATFORM	    = xilinx_vcu1525_dynamic_5_1
#PLATFORM           = /home/tbollaer/aws-fpga/SDAccel/aws_platform/xilinx_aws-vu9p-f1_4ddr-xpr-2pr_4_0/xilinx_aws-vu9p-f1_4ddr-xpr-2pr_4_0.xpfm
PLATFORM 	    = /home/tbollaer/aws-fpga-v137/SDAccel/aws_platform/xilinx_aws-vu9p-f1_dynamic_5_0/xilinx_aws-vu9p-f1_dynamic_5_0.xpfm

APP_SOURCE_FILES    = ./src/host/*.cpp 
APP_HEADER_FILES    = ./src/host/*.h 
KERNEL_SOURCE_FILES = ./src/kernel/*.cpp
KERNEL_HEADER_FILES = ./src/kernel/*.h

sw_emu_XOCC_OPTIONS = -I./src/kernel -O3
#hw_emu_XOCC_OPTIONS = -I./src/kernel -g --xp param:compiler.hlsDataflowStrictMode=warning
#hw_emu_XOCC_OPTIONS = -I./src/kernel -g --profile_kernel data:all:all:all --xp param:compiler.hlsDataflowStrictMode=warning
#hw_emu_XOCC_OPTIONS = -I./src/kernel -g --profile_kernel data:all:all:all --profile_kernel stall:all:all:all --xp param:compiler.hlsDataflowStrictMode=warning
hw_emu_XOCC_OPTIONS = -I./src/kernel -g --profile_kernel data:all:all:all --profile_kernel stall:all:all:all 
hw_XOCC_OPTIONS     = ${hw_emu_XOCC_OPTIONS}

APP		    = Filter2D.exe

# -----------------------------------------------------------------------------

XO_FILE	= xclbin/${KERNEL}.${TARGET}.xo
XCLBIN_FILE = xclbin/fpga.${NKERNEL}k.${TARGET}.xclbin

# -----------------------------------------------------------------------------

TARGET ?= hw_emu

# Compile and build the host application (.exe file)
${APP}: ${APP_SOURCE_FILES} ${APP_HEADER_FILES}
	xcpp -O2 -o $@ ${APP_SOURCE_FILES} \
	-I${XILINX_SDX}/include -I${XILINX_SDX}/runtime/include/1_2 -L${XILINX_SDX}/runtime/lib/x86_64 -L${XILINX_SDX}/lib/lnx64.o -lOpenCL -pthread \
	-fopenmp -Wl,--as-needed -Wl,-rpath,${XILINX_SDX}/lnx64/tools/opencv -L${XILINX_SDX}/lnx64/tools/opencv -lopencv_core -lopencv_highgui 

# Compile the kernel (.xo file)
${XO_FILE}: ${KERNEL_SOURCE_FILES} ${KERNEL_HEADER_FILES}
	mkdir -p xclbin
	xocc -c -t ${TARGET} ${${TARGET}_XOCC_OPTIONS} -s --platform ${PLATFORM} -k ${KERNEL} ${KERNEL_SOURCE_FILES} -o $@ 
	-firefox --new-tab _x/reports/${KERNEL}.${TARGET}/xocc_compile_${KERNEL}.${TARGET}_guidance.html &

# Link the FPGA binary (.xclbin file)
${XCLBIN_FILE}: ${XO_FILE}
	xocc -l -t ${TARGET} ${${TARGET}_XOCC_OPTIONS} -s --platform ${PLATFORM} --nk ${KERNEL}:${NKERNEL} $< -o $@ 

# Build both the host application and XCLBIN
build: ${XCLBIN_FILE} ${APP} 


F = 3;
I = ./img/test.bmp

# Execute the accelerated application
check: ${XCLBIN_FILE} emconfig.json ${APP} 
	XCL_EMULATION_MODE=${TARGET} ./${APP} -x ${XCLBIN_FILE} -i $I -f $F

# Create the emulation config file
emconfig.json:
	emconfigutil --platform ${PLATFORM} --nd 1

# -------------------------------------------------

xo: ${XO_FILE}

profile:
#	sda2protobuf sdaccel_profile_summary.csv
#	sda2wdb sdaccel_timeline_trace.csv	
	sdx_analyze trace sdaccel_timeline_trace.csv
	sdx_analyze profile sdaccel_profile_summary.csv
	sdx_analyze profile sdaccel_profile_summary.csv -f html

# -------------------------------------------------

clean:
	rm -rf ${APP} xclbin _x* sdaccel_* emconfig.json emulation* xsim* *.wcfg *.wdb .Xil prj sdx_* *.log
	
