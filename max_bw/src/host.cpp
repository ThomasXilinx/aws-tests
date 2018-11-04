#include <vector>
#include <chrono>

#include "xcl2.hpp"
#include "logger.h"
#include "cmdlineparser.h" 

using namespace sda;
using namespace sda::utils;


struct request_c 
{
    std::vector<cl::Event> wr_event;
    std::vector<cl::Event> ex_event;
    std::vector<cl::Event> rd_event;
    std::vector<int,aligned_allocator<int>> src;
    std::vector<int,aligned_allocator<int>> dst;
	cl_mem_ext_ptr_t src_ext;
	cl_mem_ext_ptr_t dst_ext;
	size_t nbytes;

    request_c () : wr_event(1), ex_event(1), rd_event(1) {}

    void init (int vector_length, int v) {
	  	auto timer_1 = std::chrono::high_resolution_clock::now();
    	src.clear(); 
    	src.resize(vector_length, v);
	    dst.clear(); 
	    dst.resize(vector_length, 0);
	  	auto timer_2 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> duration = timer_2 - timer_1;
		std::cout << "  init duration : " << duration.count()*1000 << " ms"; 

		src_ext.flags = XCL_MEM_DDR_BANK0;
		src_ext.obj   = src.data();
		src_ext.param = 0;
		dst_ext.flags = XCL_MEM_DDR_BANK1;
		dst_ext.obj   = dst.data();
		dst_ext.param = 0;   
    	nbytes = vector_length * sizeof(int);
    }
};

int main(int argc, char** argv)
{
	std::cout << std::endl;	
	std::cout << "Copy Kernel Example\n";
		
	// Parse command line
	CmdLineParser parser;
	parser.addSwitch("--fpga"   , "-x", "FPGA binary (xclbin) file to use", "xclbin/binary_container_1.awsxclbin");
	parser.addSwitch("--nruns"  , "-i", "Number of time the kernel should be called", "1");
	parser.addSwitch("--nvalues", "-n", "Number of 512-bit values to send to the kernel", "1024");
	parser.addSwitch("--latency", "-l", "Kernel latency", "128");
	parser.addSwitch("--maxreqs", "-r", "Maximum number of scheduled requests", "4");
	parser.parse(argc, argv);
	string    xclbin  = parser.value("fpga");
	unsigned  nruns   = parser.value_to_int("nruns");
	unsigned  nvalues = parser.value_to_int("nvalues");
	unsigned  latency = parser.value_to_int("latency");
	unsigned  maxreqs = parser.value_to_int("maxreqs");

	std::cout << std::endl;	
	std::cout << "FPGA binary            : " << xclbin   << std::endl;
	std::cout << "Number of runs         : " << nruns    << std::endl;
	std::cout << "Number of values       : " << nvalues  << std::endl;
	std::cout << "Kernel latency         : " << latency  << std::endl;
	std::cout << "Max scheduled requests : " << maxreqs  << std::endl;
	std::cout << std::endl;	

    //Create OpenCL CommandQueue and Kernel
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];
    cl::Context context(device);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
    //std::string device_name = device.getInfo<CL_DEVICE_NAME>(); 
    cl::Program::Binaries bins = xcl::import_binary_file(xclbin);
    //devices.resize(1);
    cl::Program program(context, {device}, bins);
    cl::Kernel krnl(program,"CopyKernel");

    // Allocate memory in host memory
    size_t vector_length = 16 * nvalues; // There are 16 int in a 512-bit word
    size_t vector_size_bytes = vector_length * sizeof(int);
    std::vector<request_c> request(maxreqs);

  	auto fpga_begin = std::chrono::high_resolution_clock::now();

  	// Run the FPGA design
    for (int n=0; n<nruns; n++) 
    {
        int  idx   = (n%maxreqs);

        // Wait for previous transaction with same idx to complete
        if (n>=maxreqs) request[idx].rd_event[0].wait();
    
        // std::cout << "Starting run n." << n << std::endl;

        // Initialize host memory data
		request[idx].init(vector_length, n);

	    // Allocate buffers in device global memory
	    cl::Buffer buffer_src(context, CL_MEM_USE_HOST_PTR | CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_ONLY,  request[idx].nbytes, &request[idx].src_ext);
	    cl::Buffer buffer_dst(context, CL_MEM_USE_HOST_PTR | CL_MEM_EXT_PTR_XILINX | CL_MEM_WRITE_ONLY, request[idx].nbytes, &request[idx].dst_ext);

	    // Copy input data to device global memory
	    q.enqueueMigrateMemObjects({buffer_src}, 0, NULL, &request[idx].wr_event[0]); // 0 means from host

	    // Set kernel arguments
	    int nargs=0;
	    krnl.setArg(nargs++, buffer_src);
	    krnl.setArg(nargs++, buffer_dst);
	    krnl.setArg(nargs++, nvalues);
	    krnl.setArg(nargs++, latency);

	    // Launch the kernel
	    q.enqueueTask(krnl, &request[idx].wr_event, &request[idx].ex_event[0]);

	    // Copy results from device global memory to host local memory
	    q.enqueueMigrateMemObjects({buffer_dst}, CL_MIGRATE_MEM_OBJECT_HOST, &request[idx].ex_event, &request[idx].rd_event[0]);
	}
	// Wait for all outstanding transactions to finish
	q.finish();

  	auto fpga_end = std::chrono::high_resolution_clock::now();

	// Report performance (if not running in emulation mode)
	if (getenv("XCL_EMULATION_MODE") == NULL) {
		std::chrono::duration<double> fpga_duration = fpga_end - fpga_begin;
		double fpga_mega_bytes = nruns*vector_size_bytes / (1024.0*1024.0);
		double fpga_throughput = fpga_mega_bytes / fpga_duration.count() ;
		std::cout << "Total data transfered : " << fpga_mega_bytes       << " MBytes"   << std::endl;
		std::cout << "Total duration        : " << fpga_duration.count() << " s"        << std::endl;
		std::cout << "Total throughput      : " << fpga_throughput       << " MBytes/s" << std::endl;
	}
	    
    // Compare the results of the Device to the simulation
    std::cout << "Validating results" << std::endl;
    int err = 0;
    for (int i=0; i<maxreqs; i++) {
    	#if 0
	    for (int j= 0; j<vector_length; j++) {
	        if (request[i].dst[j] != request[i].src[j]) {
	            std::cout << "Error: Result mismatch" << std::endl;
	            std::cout << "Req " << i << " @ " << i << " obtained = " << request[i].dst[j] << " expected = " << request[i].dst[j] << std::endl;
	            err = 1;
	            break;
	        }
	    }
	    #else
		if (request[i].dst != request[i].src) {
            std::cout << "Error: Result mismatch @ " << i << std::endl;
            err = 1;
            break;
        }
        #endif	    
    }

    std::cout << "TEST " << (err ? "FAILED" : "PASSED") << std::endl; 
    return (err ? EXIT_FAILURE :  EXIT_SUCCESS);
}
