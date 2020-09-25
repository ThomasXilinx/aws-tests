
TARGET ?= hw

# -----------------------------------------------------------------------------

OUT_DIR      = bin.${TARGET}
TMP_DIR      = tmp.${TARGET}
RPT_DIR      = rpt.${TARGET}

XO           = ${TMP_DIR}/vaddkernel.xo
XCLBIN       = ${OUT_DIR}/fpgabinary.xclbin
EXE          = host.exe

VPP_OPTIONS  = --target ${TARGET}
VPP_OPTIONS += --platform ${AWS_PLATFORM}
VPP_OPTIONS += --profile_kernel data:all:all:all --profile_kernel stall:all:all:all
VPP_OPTIONS += --save-temps --temp_dir ${TMP_DIR} --report_dir ${RPT_DIR}

GCC_OPTIONS  = -I${XILINX_XRT}/include -std=c++11 -O3
GCC_OPTIONS += -L${XILINX_XRT}/lib -lOpenCL -lpthread -lrt

# -----------------------------------------------------------------------------

all: ${XCLBIN} ${EXE}
xclbin: ${XCLBIN}
exe: ${EXE}

# -----------------------------------------------------------------------------


# Compile and build the host application (.exe file)
${EXE}: ./src/host.cpp ./src/xcl2.cpp
	mkdir -p ${OUT_DIR}
	g++ -o $@ ./src/host.cpp ./src/xcl2.cpp ${GCC_OPTIONS}

# Compile the kernel (.xo file)
${XO}: ./src/vadd.cpp
	mkdir -p ${TMP_DIR}
	v++ -c ${VPP_OPTIONS} -k vadd ./src/vadd.cpp -o $@ 

# Link the FPGA binary (.xclbin file)
${XCLBIN}: ${XO}
	mkdir -p ${OUT_DIR}
	v++ -l ${VPP_OPTIONS} $< -o $@ 

clean:
	rm -rf ${TMP_DIR} ${RPT_DIR} ${EXE} *.log *.csv *.Xil
