/**********
Copyright (c) 2017, Xilinx, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/
#include "CL/opencl.h"
#include <vector>
#include <math.h>
#include <chrono>
#include <utility>
#include <assert.h>
#include <omp.h>
#include <string.h>
#include <iostream>
#include <algorithm>

#include "cmdlineparser.h"
#include "logger.h"

using namespace sda;
using namespace sda::utils;

typedef short int16_t;
typedef unsigned short uint16_t;

void idctSoft(const int16_t block[64], const uint16_t q[64], int16_t outp[64], bool ignore_dc);

/* *************************************************************************** 

aligned_allocator

This struct provides an 4k alligned memory allocator. Using this
allocator allows data objects to be aligned for efficient data
transfer to the kernel.

The struct provides an allocate and deallocate function

*************************************************************************** */
template <typename T>
struct aligned_allocator
{
  using value_type = T;
  T* allocate(std::size_t num)
  {
    void* ptr = nullptr;
    if (posix_memalign(&ptr,4096,num*sizeof(T)))
      throw std::bad_alloc();
    return reinterpret_cast<T*>(ptr);
  }
  void deallocate(T* p, std::size_t num)
  {
    free(p);
  }
};

/* *************************************************************************** 

smalloc

Simple helper function to malloc memory of a specifc size. The
function will throw an error if the memory can not be successfully
allocated.

*************************************************************************** */
static void* smalloc(size_t size) {
  void* ptr;

  ptr = malloc(size);

  if (ptr == NULL) {
    printf("Error: Cannot allocate memory\n");
    exit(EXIT_FAILURE);
  }

  return ptr;
}

/* *************************************************************************** 

load_file_to_memory

This function reads from the file (filename) an xclbin into
memory. This binary information is returned in the argument result.

*************************************************************************** */
static int load_file_to_memory(const char *filename, char **result) {
  unsigned int size;

  FILE *f = fopen(filename, "rb");
  if (f == NULL) {
    *result = NULL;
    printf("Error: Could not read file %s\n", filename);
    exit(EXIT_FAILURE);
  }

  fseek(f, 0, SEEK_END);
  size = ftell(f);
  fseek(f, 0, SEEK_SET);

  *result = (char *) smalloc(sizeof(char)*(size+1));

  if (size != fread(*result, sizeof(char), size, f)) {
    free(*result);
    printf("Error: read of kernel failed\n");
    exit(EXIT_FAILURE);
  }

  fclose(f);
  (*result)[size] = 0;

  return size;
}


/* *************************************************************************** 

idctDriver

This class provides methods for the host application to interact with an IDCT 
compute unit in the FPGA device. All the OpenCL buffer management is handled
in this class.

The 'enqueueProcessing' method is used to enqueue an entire write/execute/read 
transaction request to the IDCT kernel. This method can safely be called
multiple times. 

The 'waitForCompletion' method waits until the compute unit is done processing
and results have been migrated back to the host. 

*************************************************************************** */

typedef std::vector<int16_t, aligned_allocator<int16_t>>   idctBatch_t;
typedef std::vector<uint16_t, aligned_allocator<uint16_t>> idctQuant_t;

class idctDriver {

public:
	idctDriver(
			  cl_context       context, 
			  cl_device_id     device, 
			  cl_kernel        krnl, 
			  cl_command_queue q) 
	{
	  mContext = context;
	  mDevice  = device;
	  mKernel  = krnl;
	  mQ       = q;

	  // DDR bank mapping
	  mBlockExt.flags = XCL_MEM_DDR_BANK0;
	  mQExt.flags     = XCL_MEM_DDR_BANK0;
	  mOutExt.flags   = XCL_MEM_DDR_BANK1;

	  mBlockExt.obj   = nullptr;
	  mBlockExt.param = 0;

	  mQExt.obj       = nullptr; 
	  mQExt.param     = 0;

	  mOutExt.obj     = nullptr; 
	  mOutExt.param   = 0;
	  
	  // Indicates whether a request has been enqueued but hasn't finished yet
	  mIsActive       = false;
	}

  void enqueueProcessing(
	    idctBatch_t *blocks,
	    idctQuant_t *q,
	    idctBatch_t *out,
	    bool         ignore_dc )
  {
	  cl_int err;

	  // If a previous request has not completed yet, wait until it finishes
	  // and results are migrated back to the host
	  waitForCompletion();

	  mIsActive = true;

    int mDevIgnoreDC        = ignore_dc ? 1 : 0;
    unsigned int mBatchSize = blocks->size()/64;

	  // Move Buffer over input vector
	  mBlockExt.obj = blocks->data(); 
	  mInBuffer[0] = clCreateBuffer(mContext, 
					CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
					blocks->size()*sizeof(int16_t), 
					&mBlockExt,
					&err);

    mQExt.obj = q->data();
	  mInBuffer[1] = clCreateBuffer(mContext, 
					CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
					q->size()*sizeof(uint16_t), 
					&mQExt,
					&err);
	  
	  // Move Buffer over output vector
	  mOutExt.obj = out->data(); 
	  mOutBuffer[0] = clCreateBuffer(mContext, 
					CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
					out->size()*sizeof(int16_t), 
					&mOutExt,
					&err);
	  
	  // Schedule writing of input data
	  clEnqueueMigrateMemObjects(mQ, 2, mInBuffer, 0, 0, nullptr, &mWriteEvent);
	  
	  // Set the kernel arguments
	  clSetKernelArg(mKernel, 0, sizeof(cl_mem), &mInBuffer[0]);
	  clSetKernelArg(mKernel, 1, sizeof(cl_mem), &mInBuffer[1]);
	  clSetKernelArg(mKernel, 2, sizeof(cl_mem), &mOutBuffer[0]);
	  clSetKernelArg(mKernel, 3, sizeof(int),    &mDevIgnoreDC);
	  clSetKernelArg(mKernel, 4, sizeof(unsigned int), &mBatchSize);

	  // Schedule execution of kernel
	  clEnqueueTask(mQ, mKernel, 1, &mWriteEvent, &mRunEvent);

	  // Schedule reading of output results
	  clEnqueueMigrateMemObjects(mQ, 1, mOutBuffer, CL_MIGRATE_MEM_OBJECT_HOST, 1, &mRunEvent, &mReadEvent);
	}  

  void waitForCompletion()
  {
	  // If a previous request has not completed yet, wait until it finishes
	  // and results are migrated back to the host, then release OpenCL buffers
	  if (mIsActive == true) 
    {
      clWaitForEvents(1, &mReadEvent);

      clReleaseMemObject( mOutBuffer[0]);
      clReleaseMemObject( mInBuffer[0]);
      clReleaseMemObject( mInBuffer[1]);

      clReleaseEvent(mWriteEvent);
      clReleaseEvent(mRunEvent);
      clReleaseEvent(mReadEvent);

      mIsActive = false;
    }
  };

  ~idctDriver() {	
  	// If necessary, there is still an unfinished request,
    // for it to complete and release the OpenCL buffers and events
  	waitForCompletion();
  }  

private:
  cl_context        mContext;
  cl_device_id      mDevice;
  cl_kernel         mKernel;
  cl_command_queue  mQ;

  bool              mIsActive;

  cl_mem            mInBuffer[2];
  cl_mem            mOutBuffer[1];

  cl_mem_ext_ptr_t  mBlockExt;
  cl_mem_ext_ptr_t  mQExt;
  cl_mem_ext_ptr_t  mOutExt;

  cl_event          mWriteEvent;
  cl_event          mRunEvent;
  cl_event          mReadEvent;
};


/* *************************************************************************** 

idctDispatcher

This class provides a service for managing multiple outstanding requests
to the IDCT kernel. The 'numSched' constructor argument is used to specify 
the maximum number of outstanding requests allowed.

The 'run' method circles through a vector of idctDriver 

*************************************************************************** */

class idctDispatcher {

public:

  idctDispatcher(
    	cl_context       context, 
      cl_device_id     device, 
      cl_kernel        krnl, 
      cl_command_queue q,
      unsigned int     numSched )
  {
  	// Populate a vector of 'idctDriver' objects with 'numSched' elements
  	for (int i=0; i<numSched; i++) {
  	  mRequest.push_back( idctDriver(context, device, krnl, q) );
    }

  	mCount       = 0;
  	mNumSched    = numSched;
  	mQ           = q;
  }

  void run(
      idctBatch_t *blocks,
      idctQuant_t *q,
      idctBatch_t *out,
      bool         ignore_dc )
  {
  	// Circle through the vector of idctDrivers and enqueue processing requests,
  	// waiting for completion if the previous transaction hasn't finished yet
  	mRequest[mCount%mNumSched].enqueueProcessing(blocks, q, out, ignore_dc);
  	mCount++;
  }

  void finish()
  {
  	// Wait until all enqueued transactions in 'mQ' complete
  	clFinish(mQ);
  }

  ~idctDispatcher() 
  {
  	// std::vectors are automatically destroyed
  };

private:
  std::vector<idctDriver> mRequest;
  unsigned int            mCount;
  unsigned int            mNumSched;
  cl_command_queue        mQ;

};


/* *************************************************************************** 

runFPGA

This function guides the kernel execution of the idct algorithm.

*************************************************************************** */
void runFPGA(
      std::vector<idctBatch_t>   &block,
      idctQuant_t                &q,
      std::vector<idctBatch_t>   &out,
      bool                        ignore_dc,
      idctDispatcher             &dispatcher )
{
  // Get the number of batches
  size_t numBatch = block.size();        

  // Dispatch as many IDCT requests as there are batches to process
  for(size_t i = 0; i < numBatch; i++) 
  {
    dispatcher.run(&block[i], &q, &out[i], ignore_dc);
  }

  // Wait until all dispatched jobs have completed
  dispatcher.finish();
}

/* *************************************************************************** 

runCPU

This function performs the host code computation of the idct
algorithm.

*************************************************************************** */
void runCPU(
      std::vector<idctBatch_t>   &block,
      idctQuant_t                &q,
      std::vector<idctBatch_t>   &out,
	    bool                        ignore_dc )
{
  size_t numBatch = block.size();          
  for(size_t i = 0; i < numBatch; i++)     // for each batch
  {
    size_t sizBatch = block[i].size()/64;  
    for (size_t j = 0; j < sizBatch; j++ ) // for each block in a batch 
    {
      // Run the reference IDCT implementation
      idctSoft(&block[i][j*64], &q[0], &out[i][j*64], ignore_dc);
    }
  }
}


/* *************************************************************************** 

main

This function is the main function of the idct program. It illustrates
the basic opencl hostcode setup, followed by the idct execution on
host (CPU) and an accelerated flow (FPGA). With a functional
comparison between host and fpga exectuion.

*************************************************************************** */
int main(int argc, char* argv[]) {

  CmdLineParser parser;
  parser.addSwitch("--fpga",      "-x", "FPGA binary file to use (xclbin)", "binary_container_1.xclbin");
  parser.addSwitch("--numbatch",  "-n", "Number of batches to be processed by the application", "8192");
  parser.addSwitch("--batchsize", "-b", "Number of 8x8 blocks processed by one kernel invocation", "512");
  parser.addSwitch("--numsched",  "-s", "Number of scheduled transactions", "1");

  // Parse all command line options
  parser.parse(argc, argv);
  string fpgaBinary  = parser.value("fpga");
  size_t numBatch    = parser.value_to_int("numbatch");
  size_t sizBatch    = parser.value_to_int("batchsize");
  int    numSched    = parser.value_to_int("numsched");


  // Limit number of blocks and batch size for emulation modes
  char *xcl_mode = getenv("XCL_EMULATION_MODE");
  if (xcl_mode != NULL) {
    numBatch = min(numBatch, (size_t)  4);
    sizBatch = min(sizBatch, (size_t)256);
  }

  // Total number of block processed by the application
  size_t totBlocks = numBatch * sizBatch;

  std::cout << std::endl;
  std::cout << "FPGA binary file                  : " << fpgaBinary << std::endl;
  std::cout << "Number of batches to be processed : " << numBatch   << std::endl;
  std::cout << "Number of IDCT blocks per batch   : " << sizBatch   << " / " <<  sizBatch*128/(1024.0*1024.0) << " MB" << std::endl;
  std::cout << "Total number of blocks processed  : " << totBlocks  << " / " << totBlocks*128/(1024.0*1024.0) << " MB" << std::endl;
  std::cout << "Number of scheduled transactions  : " << numSched   << std::endl;
  std::cout << std::endl;


  // *********** Allocate and initialize test vectors **********

  bool ignore_dc = true;
  
  // Allocate storage
  idctQuant_t source_q(64);
  std::vector<idctBatch_t>   source_block(numBatch);
  std::vector<idctBatch_t>   golden_vpout(numBatch);
  std::vector<idctBatch_t>   result_vpout(numBatch);

  for(size_t i = 0; i < numBatch; i++) {
    source_block[i].resize(64*sizBatch);
    golden_vpout[i].resize(64*sizBatch);
    result_vpout[i].resize(64*sizBatch);
  }

  // Initialize input blocks
  unsigned cnt = 0;
  for(size_t i = 0; i < numBatch; i++) {
    for(size_t j = 0; j < 64*sizBatch; j++) {
      source_block[i][j] = cnt++;
    }
  }
	
  // Initialize quantization table
  for(size_t j = 0; j < 64; j++) {
    source_q[j] = j;
  }

  // *********** OpenCL Host Code Setup **********

  // Connect to first platform
  int err;
  char cl_platform_vendor[1001];
  char cl_platform_name[1001];
  char cl_device_name[1001];

  cl_platform_id platform_id;         // platform id
  cl_device_id device_id;             // compute device id
  cl_context context;                 // compute context

  // Get number of platforms
  cl_uint platform_count;
  clGetPlatformIDs(0, nullptr, &platform_count);

  // get all platforms
  std::vector<cl_platform_id> platforms(platform_count);
  clGetPlatformIDs(platform_count, platforms.data(), nullptr);

  bool found = false;
  for (int p = 0; p < (int)platform_count; ++p) {  
    platform_id = platforms[p];
    clGetPlatformInfo(platform_id,CL_PLATFORM_VENDOR,1000,(void *)cl_platform_vendor,NULL);
    clGetPlatformInfo(platform_id,CL_PLATFORM_NAME  ,1000,(void *)cl_platform_name,NULL);
    if(!strcmp(cl_platform_vendor,"Xilinx")) {
      found = true;
      break;
    }
  }
  if (!found){
    std::cout << "Platform Not Found\n";
    return err;
  }

  err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ACCELERATOR, 1, &device_id, NULL);
  if (err != CL_SUCCESS) {
    std::cout << "FAILED TEST - Device\n";
    return err;
  }
  
  context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
  if (!context || (err != CL_SUCCESS)) {
    std::cout << "FAILED TEST - Context \n";
    return err;
  }
  
  clGetDeviceInfo(device_id, CL_DEVICE_NAME, 1000, (void*)cl_device_name, NULL);

  std::cout << "Found device: " << cl_device_name << std::endl;

  std::cout << "Loading FPGA binary: " << fpgaBinary << std::endl; 
  char *krnl_bin;
  size_t krnl_size;
  krnl_size = load_file_to_memory(fpgaBinary.c_str(), &krnl_bin);

  cl_program program = clCreateProgramWithBinary(context, 1,
						 (const cl_device_id* ) &device_id, &krnl_size,
						 (const unsigned char**) &krnl_bin,
						 NULL, &err);

  // Create Kernel
  std::cout << "Creating kernel: krnl_idct" << std::endl;
  cl_kernel krnl = clCreateKernel(program, "krnl_idct", &err);

  // Create Command Queue
  cl_command_queue q = clCreateCommandQueue(context, device_id, 
					    CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);

  // Create dispatcher of FPGA processing requests
  idctDispatcher dispatcher(context, device_id, krnl, q, numSched);

  std::cout << "Setup complete" << std::endl;
  std::cout << std::endl;

  // *********** Host (CPU) execution **********
  std::cout << "Running CPU version" << std::endl;
  auto cpu_begin = std::chrono::high_resolution_clock::now();
  runCPU(source_block, source_q, golden_vpout, ignore_dc);
  auto cpu_end = std::chrono::high_resolution_clock::now();

  // *********** Accelerator execution **********
  std::cout << "Running FPGA version" << std::endl;
  auto fpga_begin = std::chrono::high_resolution_clock::now();
  runFPGA(source_block, source_q, result_vpout, ignore_dc, dispatcher);
  auto fpga_end = std::chrono::high_resolution_clock::now();

  // *********** OpenCL Host Code cleanup **********
  clReleaseCommandQueue(q);
  clReleaseKernel(krnl);
  clReleaseProgram(program);
  clReleaseContext(context);

  // *********** Comparison (Host to Acceleration)  **********
  std::cout << std::endl;
  std::cout << "Validating results" << std::endl;
  int krnl_err = 0;
  for(size_t i = 0; i < numBatch; i++)
  {
    if(result_vpout[i] != golden_vpout[i])
    {
      printf("Error: Result mismatch\n");
      krnl_err = 1;
      break;
    } 
  }
  std::cout << "TEST " << (krnl_err ? "FAILED" : "PASSED") << std::endl;
  std::cout << std::endl;

  // *********** Computational Statistics  **********
  // Only reported in the HW execution mode as wall 
  // clock time is meaningless in emulation.
  if (xcl_mode == NULL) {
    std::cout << "Performance statistics" << std::endl;
    std::chrono::duration<double> cpu_duration = cpu_end - cpu_begin;
    std::chrono::duration<double> fpga_duration = fpga_end - fpga_begin;

    std::cout << "  CPU Time:        " << cpu_duration.count() << " s" << std::endl;
    std::cout << "  CPU Throughput:  " 
	      << (double) totBlocks*128 / cpu_duration.count() / (1024.0*1024.0)
	      << " MB/s" << std::endl;
    std::cout << "  FPGA Time:       " << fpga_duration.count() << " s" << std::endl;
    std::cout << "  FPGA Throughput: " 
	      << (double) totBlocks*128 / fpga_duration.count() / (1024.0*1024.0)
	      << " MB/s" << std::endl;
    std::cout << std::endl;
  } 

  return (krnl_err ? EXIT_FAILURE : EXIT_SUCCESS);
}



/* *************************************************************************** 

idctSoft

Original software implementation of IDCT algorithm used to generate
golden reference data.

*************************************************************************** */
void idctSoft(
        const int16_t block[64], 
	      const uint16_t q[64], 
	      int16_t outp[64], 
	      bool ignore_dc) 
{
  int32_t intermed[64];

  const uint16_t w1 = 2841; // 2048*sqrt(2)*cos(1*pi/16)
  const uint16_t w2 = 2676; // 2048*sqrt(2)*cos(2*pi/16)
  const uint16_t w3 = 2408; // 2048*sqrt(2)*cos(3*pi/16)
  const uint16_t w5 = 1609; // 2048*sqrt(2)*cos(5*pi/16)
  const uint16_t w6 = 1108; // 2048*sqrt(2)*cos(6*pi/16)
  const uint16_t w7 = 565;  // 2048*sqrt(2)*cos(7*pi/16)

  const uint16_t w1pw7 = w1 + w7;
  const uint16_t w1mw7 = w1 - w7;
  const uint16_t w2pw6 = w2 + w6;
  const uint16_t w2mw6 = w2 - w6;
  const uint16_t w3pw5 = w3 + w5;
  const uint16_t w3mw5 = w3 - w5;

  const uint16_t r2 = 181; // 256/sqrt(2)

  // Horizontal 1-D IDCT.
  for (int y = 0; y < 8; ++y) 
  {
    int y8 = y * 8;
    int32_t x0 = (((ignore_dc && y == 0) ? 0 : (block[y8 + 0] * q[y8 + 0]) << 11)) + 128;
    int32_t x1 = (block[y8 + 4] * q[y8 + 4]) << 11;
    int32_t x2 = block[y8 + 6] * q[y8 + 6];
    int32_t x3 = block[y8 + 2] * q[y8 + 2];
    int32_t x4 = block[y8 + 1] * q[y8 + 1];
    int32_t x5 = block[y8 + 7] * q[y8 + 7];
    int32_t x6 = block[y8 + 5] * q[y8 + 5];
    int32_t x7 = block[y8 + 3] * q[y8 + 3];

    // If all the AC components are zero, then the IDCT is trivial.
    if (x1 ==0 && x2 == 0 && x3 == 0 && x4 == 0 && x5 == 0 && x6 == 0 && x7 == 0) 
    {
      int32_t dc = (x0 - 128) >> 8; // coefficients[0] << 3
      intermed[y8 + 0] = dc;
      intermed[y8 + 1] = dc;
      intermed[y8 + 2] = dc;
      intermed[y8 + 3] = dc;
      intermed[y8 + 4] = dc;
      intermed[y8 + 5] = dc;
      intermed[y8 + 6] = dc;
      intermed[y8 + 7] = dc;
      continue;
    }
        
    // Prescale.
        
    // Stage 1.
    int32_t x8 = w7 * (x4 + x5);
    x4 = x8 + w1mw7*x4;
    x5 = x8 - w1pw7*x5;
    x8 = w3 * (x6 + x7);
    x6 = x8 - w3mw5*x6;
    x7 = x8 - w3pw5*x7;
        
    // Stage 2.
    x8 = x0 + x1;
    x0 -= x1;
    x1 = w6 * (x3 + x2);
    x2 = x1 - w2pw6*x2;
    x3 = x1 + w2mw6*x3;
    x1 = x4 + x6;
    x4 -= x6;
    x6 = x5 + x7;
    x5 -= x7;
        
    // Stage 3.
    x7 = x8 + x3;
    x8 -= x3;
    x3 = x0 + x2;
    x0 -= x2;
    x2 = (r2*(x4+x5) + 128) >> 8;
    x4 = (r2*(x4-x5) + 128) >> 8;
        
    // Stage 4.
    intermed[y8+0] = (x7 + x1) >> 8;
    intermed[y8+1] = (x3 + x2) >> 8;
    intermed[y8+2] = (x0 + x4) >> 8;
    intermed[y8+3] = (x8 + x6) >> 8;
    intermed[y8+4] = (x8 - x6) >> 8;
    intermed[y8+5] = (x0 - x4) >> 8;
    intermed[y8+6] = (x3 - x2) >> 8;
    intermed[y8+7] = (x7 - x1) >> 8;
  }
    
  // Vertical 1-D IDCT.
  for (int32_t x = 0; x < 8; ++x) 
  {
    // Similar to the horizontal 1-D IDCT case, if all the AC components are zero, then the IDCT is trivial.
    // However, after performing the horizontal 1-D IDCT, there are typically non-zero AC components, so
    // we do not bother to check for the all-zero case.
        
    // Prescale.
    int32_t y0 = (intermed[8*0+x] << 8) + 8192;
    int32_t y1 = intermed[8*4+x] << 8;
    int32_t y2 = intermed[8*6+x];
    int32_t y3 = intermed[8*2+x];
    int32_t y4 = intermed[8*1+x];
    int32_t y5 = intermed[8*7+x];
    int32_t y6 = intermed[8*5+x];
    int32_t y7 = intermed[8*3+x];
        
    // Stage 1.
    int32_t y8 = w7*(y4+y5) + 4;
    y4 = (y8 + w1mw7*y4) >> 3;
    y5 = (y8 - w1pw7*y5) >> 3;
    y8 = w3*(y6+y7) + 4;
    y6 = (y8 - w3mw5*y6) >> 3;
    y7 = (y8 - w3pw5*y7) >> 3;
        
    // Stage 2.
    y8 = y0 + y1;
    y0 -= y1;
    y1 = w6*(y3+y2) + 4;
    y2 = (y1 - w2pw6*y2) >> 3;
    y3 = (y1 + w2mw6*y3) >> 3;
    y1 = y4 + y6;
    y4 -= y6;
    y6 = y5 + y7;
    y5 -= y7;
        
    // Stage 3.
    y7 = y8 + y3;
    y8 -= y3;
    y3 = y0 + y2;
    y0 -= y2;
    y2 = (r2*(y4+y5) + 128) >> 8;
    y4 = (r2*(y4-y5) + 128) >> 8;
        
    // Stage 4.
    outp[8*0+x] = (y7 + y1) >> 11;
    outp[8*1+x] = (y3 + y2) >> 11;
    outp[8*2+x] = (y0 + y4) >> 11;
    outp[8*3+x] = (y8 + y6) >> 11;
    outp[8*4+x] = (y8 - y6) >> 11;
    outp[8*5+x] = (y0 - y4) >> 11;
    outp[8*6+x] = (y3 - y2) >> 11;
    outp[8*7+x] = (y7 - y1) >> 11;
  }
}
