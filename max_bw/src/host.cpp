#include <vector>

#include "xcl2.hpp"
#include "logger.h"
#include "cmdlineparser.h" 

using namespace sda;
using namespace sda::utils;


int main(int argc, char** argv)
{
	std::cout << std::endl;	
	std::cout << "Copy Kernel Example\n";
		
	// ---------------------------------------------------------------------------------
	// Parse command line
	// ---------------------------------------------------------------------------------

	CmdLineParser parser;
	parser.addSwitch("--fpga"   , "-x", "FPGA binary (xclbin) file to use", "xclbin/binary_container_1.awsxclbin");
	parser.addSwitch("--nruns"  , "-i", "Number of time the kernel should be called", "1");
	parser.addSwitch("--nvalues", "-n", "Number of 512-bit values to send to the kernel", "1024");
	parser.addSwitch("--latency", "-l", "Kernel latency", "128");

	//parse all command line options
	parser.parse(argc, argv);
	string    xclbin  = parser.value("fpga");
	unsigned  nruns   = parser.value_to_int("nruns");
	unsigned  nvalues = parser.value_to_int("nvalues");
	unsigned  latency = parser.value_to_int("latency");

	std::cout << std::endl;	
	std::cout << "FPGA binary            : " << xclbin   << std::endl;
	std::cout << "Number of runs         : " << nruns    << std::endl;
	std::cout << "Number of values       : " << nvalues  << std::endl;
	std::cout << "Kernel latency         : " << latency  << std::endl;
	std::cout << std::endl;	


    //Allocate Memory in Host Memory
    size_t vector_length     = 16 * nvalues; // There are 16 int in a 512-bit word
    size_t vector_size_bytes = vector_length * sizeof(int);
    std::vector<int,aligned_allocator<int>> src(vector_length);
    std::vector<int,aligned_allocator<int>> dst(vector_length);

    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    cl::Context context(device);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
    std::string device_name = device.getInfo<CL_DEVICE_NAME>(); 

    //Create Program and Kernel
    std::string binaryFile = xclbin; //xcl::find_binary_file(device_name,"kernel");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    cl::Program program(context, devices, bins);
    cl::Kernel krnl(program,"CopyKernel");

    int val = 0;
    int err = 0;

    for (int n=0; n<nruns; n++) 
    {
        std::cout << "Starting run n." << n << std::endl;

        std::vector<cl::Event> write_event(1);
        std::vector<cl::Event> execute_event(1);
        std::vector<cl::Event> read_event(1);

	    // Initialize the test data 
	    for(int i=0; i<vector_length; i++) {
	        src[i] = val++;
	        dst[i] = 0;
	    }
	    //int err;
		cl_mem_ext_ptr_t src_ext;
		src_ext.flags = XCL_MEM_DDR_BANK0;
		src_ext.obj   = src.data();
		src_ext.param = 0;
		cl_mem_ext_ptr_t dst_ext;
		dst_ext.flags = XCL_MEM_DDR_BANK1;
		dst_ext.obj   = dst.data();
		dst_ext.param = 0;

	    // Allocate Buffer in Global Memory
	    cl::Buffer buffer_src(context, CL_MEM_USE_HOST_PTR | CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_ONLY,  vector_size_bytes, &src_ext);
	    cl::Buffer buffer_dst(context, CL_MEM_USE_HOST_PTR | CL_MEM_EXT_PTR_XILINX | CL_MEM_WRITE_ONLY, vector_size_bytes, &dst_ext);

	    // Copy input data to device global memory
	    q.enqueueMigrateMemObjects({buffer_src}, 0, NULL, &write_event[0]); // 0 means from host

	    // Set the Kernel Arguments
	    int nargs=0;
	    krnl.setArg(nargs++, buffer_src);
	    krnl.setArg(nargs++, buffer_dst);
	    krnl.setArg(nargs++, nvalues);
	    krnl.setArg(nargs++, latency);

	    // Launch the Kernel
	    q.enqueueTask(krnl, &write_event, &execute_event[0]);

	    // Copy Result from Device Global Memory to Host Local Memory
	    q.enqueueMigrateMemObjects({buffer_dst}, CL_MIGRATE_MEM_OBJECT_HOST, &execute_event, &read_event[0]);
	    q.finish();
	    
	    // Compare the results of the Device to the simulation
	    for (int i = 0 ; i < vector_length ; i++){
	        if (dst[i] != src[i]){
	            std::cout << "Error: Result mismatch" << std::endl;
	            std::cout << "@ i = " << i << " obtained = " << src[i] << " expected = " << dst[i] << std::endl;
	            err = 1;
	            break;
	        }
	    }
	}

    std::cout << "TEST " << (err ? "FAILED" : "PASSED") << std::endl; 
    return (err ? EXIT_FAILURE :  EXIT_SUCCESS);
}
