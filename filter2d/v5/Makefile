
KERNEL  	    = Filter2DKernel
NKERNEL 	    ?= 3
PLATFORM 	    ?= xilinx_aws-vu9p-f1-04261818_dynamic_5_0
#PLATFORM	    ?= xilinx_u200_xdma_201830_1
#PLATFORM	    ?= xilinx_vcu1525_dynamic_5_1

APP_SOURCE_FILES    = ./src/host/*.cpp 
APP_HEADER_FILES    = ./src/host/*.h 
KERNEL_SOURCE_FILES = ./src/kernel/*.cpp
KERNEL_HEADER_FILES = ./src/kernel/*.h

XOCC_COMP_OPTIONS += -I./src/kernel -g --profile_kernel data:all:all:all --profile_kernel stall:all:all:all 
XOCC_LINK_OPTIONS += 

APP		   = Filter2D.exe

# -----------------------------------------------------------------------------

XO_FILE	= xclbin/${KERNEL}.${TARGET}.xo
XCLBIN_FILE = xclbin/fpga.${NKERNEL}k.${TARGET}.xclbin
TMP_DIR = builddir.${NKERNEL}k.${TARGET}

# -----------------------------------------------------------------------------

TARGET ?= hw_emu


APP_COMP_OPTIONS = -I${XILINX_SDX}/include -std=c++14 
APP_LINK_OPTIONS = -I${XILINX_SDX}/runtime/include/1_2 -L${XILINX_SDX}/runtime/lib/x86_64 -L${XILINX_SDX}/lib/lnx64.o -lxilinxopencl -pthread
#APP_LINK_OPTIONS = -I${XILINX_XRT}/include/ -L${XILINX_XRT}/lib/ -lxilinxopencl -lpthread -lrt -lstdc++

# Compile and build the host application (.exe file)
${APP}: ${APP_SOURCE_FILES} ${APP_HEADER_FILES}
	xcpp -O2 -o $@ ${APP_SOURCE_FILES} ${APP_COMP_OPTIONS} ${APP_LINK_OPTIONS} \
	-fopenmp -Wl,--as-needed -Wl,-rpath,${XILINX_SDX}/lnx64/tools/opencv -L${XILINX_SDX}/lnx64/tools/opencv -lopencv_core -lopencv_highgui 

# Compile the kernel (.xo file)
${XO_FILE}: ${KERNEL_SOURCE_FILES} ${KERNEL_HEADER_FILES}
	mkdir -p xclbin
	xocc -c -t ${TARGET} ${XOCC_COMP_OPTIONS} -s --temp_dir ${TMP_DIR} --platform ${PLATFORM} -k ${KERNEL} ${KERNEL_SOURCE_FILES} -o $@ 

# Link the FPGA binary (.xclbin file)
${XCLBIN_FILE}: ${XO_FILE}
	xocc -l -t ${TARGET} ${XOCC_LINK_OPTIONS} -s --temp_dir ${TMP_DIR} --platform ${PLATFORM} --nk ${KERNEL}:${NKERNEL} $< -o $@ 

# Build both the host application and XCLBIN
build: ${XCLBIN_FILE} ${APP} 


F = 0
I = ./img/test.bmp
N = 2

# Execute the accelerated application
check: ${XCLBIN_FILE} emconfig.json ${APP} 
	XCL_EMULATION_MODE=${TARGET} ./${APP} -x ${XCLBIN_FILE} -i $I -f $F -n $N

# Create the emulation config file
emconfig.json:
	emconfigutil --platform ${PLATFORM} --nd 1

# -------------------------------------------------

xo: ${XO_FILE}

profile:
	sdx_analyze trace sdaccel_timeline_trace.csv
	sdx_analyze profile sdaccel_profile_summary.csv
	sdx_analyze profile sdaccel_profile_summary.csv -f html
	sdx -workspace w -report sdaccel_timeline_trace.wdb &

# -------------------------------------------------

all:
	make build TARGET=hw NKERNEL=1
	make build TARGET=hw NKERNEL=3
	make build TARGET=hw NKERNEL=6

clean:
	rm -rf ${APP} xclbin _x* builddir* sdaccel_* emconfig.json emulation* xsim* *.wcfg *.wdb .Xil prj sdx_* *.log .run* ./img/*_ref.bmp ./img/*_out.bmp
