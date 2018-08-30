
KERNEL  	    	:= krnl_idct
APP	   		:= app.exe

DEVICE                  ?= f1-dynamic
TARGET                  ?= hw_emu
NKERNEL                 ?= 1

XCPP_SOURCE_FILES	:= ./src/host.cpp ./src/logger.cpp ./src/cmdlineparser.cpp
XCPP_HEADER_FILES	:= 
XCPP_COMP_OPTIONS	:= -O2
XCPP_LINK_OPTIONS	:=

XOCC_SOURCE_FILES	:= ./src/krnl_idct.cpp 
XOCC_HEADER_FILES	:= 

ifeq ($(TARGET),sw_emu)
XOCC_COMP_OPTIONS 	:= -O3
else
XOCC_COMP_OPTIONS 	:= -g --profile_kernel data:all:all:all --xp param:compiler.hlsDataflowStrictMode=warning
endif

ifeq ($(DEVICE),f1-4ddr)
XOCC_LINK_OPTIONS   	:= --xp misc:map_connect=add.kernel.krnl_idct_1.M_AXI_GMEM0.core.OCL_REGION_0.M00_AXI \
			   --xp misc:map_connect=add.kernel.krnl_idct_1.M_AXI_GMEM1.core.OCL_REGION_0.M00_AXI \
			   --xp misc:map_connect=add.kernel.krnl_idct_1.M_AXI_GMEM2.core.OCL_REGION_0.M00_AXI 
else
XOCC_LINK_OPTIONS   	:= --sp krnl_idct_1.m_axi_gmem0:bank0 \
        		   --sp krnl_idct_1.m_axi_gmem1:bank0 \
        		   --sp krnl_idct_1.m_axi_gmem2:bank0 
endif	

# -----------------------------------------------------------------------------

XO_FILE			:= xclbin/${KERNEL}.${TARGET}.${DEVICE}.xo
XCLBIN_FILE 		:= xclbin/${KERNEL}.${TARGET}.${DEVICE}.xclbin

ifeq ($(DEVICE),f1-4ddr)
PLATFORM := ${AWS_FPGA_REPO_DIR}/SDAccel/aws_platform/xilinx_aws-vu9p-f1_4ddr-xpr-2pr_4_0/xilinx_aws-vu9p-f1_4ddr-xpr-2pr_4_0.xpfm
endif
ifeq ($(DEVICE),f1-dynamic)
PLATFORM := ${AWS_FPGA_REPO_DIR}/SDAccel/aws_platform/xilinx_aws-vu9p-f1-04261818_dynamic_5_0/xilinx_aws-vu9p-f1-04261818_dynamic_5_0.xpfm
endif
ifeq ($(DEVICE),vcu1525)
PLATFORM := xilinx_vcu1525_dynamic_5_1
endif

# -----------------------------------------------------------------------------

app: ${APP}

xo: ${XO_FILE}

xclbin: ${XCLBIN_FILE}	

build: ${APP} ${XCLBIN_FILE}

# Execute the accelerated application
check: ${APP} ${XCLBIN_FILE} emconfig.json
	XCL_EMULATION_MODE=${TARGET} ./${APP} -x ${XCLBIN_FILE} -n 16

# Compile and build the host application (.exe file)
${APP}: ${XCPP_SOURCE_FILES} ${XCPP_HEADER_FILES}
	xcpp -o $@ ${XCPP_SOURCE_FILES} \
	${XCPP_COMP_OPTIONS} ${XCPP_LINK_OPTIONS} \
	-I${XILINX_SDX}/include -I${XILINX_SDX}/runtime/include/1_2 \
	-L${XILINX_SDX}/runtime/lib/x86_64 -L${XILINX_SDX}/lib/lnx64.o \
	-lOpenCL -pthread

# Compile the kernel (.xo file)
${XO_FILE}: ${XOCC_SOURCE_FILES} ${XOCC_HEADER_FILES}
	mkdir -p xclbin
	xocc -c -t ${TARGET} ${XOCC_COMP_OPTIONS} -s --platform ${PLATFORM} -k ${KERNEL} ${XOCC_SOURCE_FILES} -o $@ 

# Link the FPGA binary (.xclbin file)
${XCLBIN_FILE}: ${XO_FILE}
	xocc -l -t ${TARGET} ${XOCC_COMP_OPTIONS} -s --platform ${PLATFORM} --nk ${KERNEL}:${NKERNEL} ${XOCC_LINK_OPTIONS} ${XO_FILE} -o $@  

# Create the emulation config file
emconfig.json:
	emconfigutil --platform ${PLATFORM} --nd 1

# -------------------------------------------------

profile:
	# sda2protobuf sdaccel_profile_summary.csv
	# sda2wdb sdaccel_timeline_trace.csv	
	sdx_analyze trace sdaccel_timeline_trace.csv
	sdx_analyze profile sdaccel_profile_summary.csv
	sdx_analyze profile sdaccel_profile_summary.csv -f html	

# -------------------------------------------------

clean:
	rm -rf ${APP} xclbin _x* sdaccel_* emconfig.json emulation* xsim* *.wcfg *.wdb .Xil prj sdx_* *.log consolidated.cf kernel_info.dat
	
