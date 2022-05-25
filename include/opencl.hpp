#ifndef OPENCL_H
#define OPENCL_H

// For loading shader files
#define MAX_SOURCE_SIZE (0x100000)

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <map>
#include <string>
#include <vector>

typedef struct OpenClBuffer {
    cl_mem buffer;
    size_t size;
} OpenClBuffer;

class OpenCl {
public:
    OpenCl(size_t size_x, size_t size_y, char *filename, std::vector<std::string> bufferNames, std::vector<size_t> bufferSizes, std::vector<std::string> kernelNames);
    void prepare(std::vector<std::string> bufferNames, std::vector<size_t> bufferSizes, std::vector<std::string> kernelNames);
    void setKernelArgs();
    void writeBuffer();
    void step();
    void readBuffer();
    void cleanup();

    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_context context;
    cl_command_queue command_queue;

    std::map<std::string, OpenClBuffer> buffers;

    cl_program program;
    std::map<std::string, cl_kernel> kernels;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret;

    size_t source_size;
    char *source_str;

    // Kernel size for parallelisation
    size_t global_item_size[2];
    size_t local_item_size[2];

    char *filename;
};


#endif
