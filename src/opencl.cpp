#include <stdio.h>
#include <string>
#include <vector>

#include "opencl.hpp"

using namespace std;

OpenCl::OpenCl(
    size_t size_x,
    size_t size_y,
    char *filename,
    vector<string> bufferNames,
    vector<size_t> bufferSizes, 
    vector<string> kernelNames
) {
    this->local_item_size[0] = size_x;
    this->local_item_size[1] = size_y;

    this->global_item_size[0] = size_x;
    this->global_item_size[1] = size_y;

    this->filename = filename;
    
    this->prepare(bufferNames, bufferSizes, kernelNames);
}

void OpenCl::prepare(vector<string> bufferNames, vector<size_t> bufferSizes, vector<string> kernelNames) {
    FILE *fp;
    fp = fopen(this->filename, "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    
    source_str = (char *)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);
    
    /* Get Platform/Device Information */
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);

    /* Create OpenCL Context */
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    if (ret != CL_SUCCESS)
      fprintf(stderr, "Failed on function clCreateContext: %d\n", ret);

    /* Create command queue */
    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    /* Create Buffer Object */
    for (size_t i = 0; i < bufferNames.size(); i++) {
        cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, bufferSizes[i], NULL, &ret);
        this->buffers[bufferNames[i]] = {buffer, bufferSizes[i]};
    }

    /* Create kernel program from source file*/
    program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
    if (ret != CL_SUCCESS)
      fprintf(stderr, "Failed on function clCreateProgramWithSource: %d\n", ret);
    
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (ret != CL_SUCCESS)
      fprintf(stderr, "Failed on function clBuildProgram: %d\n", ret);
    
    size_t len = 10000;
    ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
    char *buffer = (char *)calloc(len, sizeof(char));
    ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
    fprintf(stderr, "%s\n", buffer);

    /* Create data parallel OpenCL kernel */
    for (size_t i = 0; i < kernelNames.size(); i++) {
        kernels[kernelNames[i]] = clCreateKernel(program, kernelNames[i].c_str(), &ret);

        if (ret != CL_SUCCESS)
            fprintf(stderr, "Failed on function clCreateKernel: %d\n", ret);
    }
}