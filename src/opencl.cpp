#include <map>
#include <stdio.h>
#include <string>
#include <vector>

#include "opencl.hpp"

using namespace std;

OpenCl::OpenCl(
    size_t size_x,
    size_t size_y,
    char *filename,
    bool dualKernel,
    vector<string> bufferNames,
    vector<size_t> bufferSizes, 
    vector<string> kernelNames
) {
    this->local_item_size[0] = size_x;
    this->local_item_size[1] = size_y;

    this->global_item_size[0] = size_x;
    this->global_item_size[1] = size_y;

    this->filename = filename;
    this->dualKernel = dualKernel;
    
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
        if (ret != CL_SUCCESS)
            fprintf(stderr, "Failed on function clCreateBuffer for buffer %s: %d\n", bufferNames[i].c_str(), ret);
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
            fprintf(stderr, "Failed on function clCreateKernel %s: %d\n", kernelNames[i].c_str(), ret);

        if (this->dualKernel) {
            string name = string(kernelNames[i]) + "2";
            kernels[name] = clCreateKernel(program, kernelNames[i].c_str(), &ret);

            if (ret != CL_SUCCESS)
                fprintf(stderr, "Failed on function clCreateKernel %s: %d\n", kernelNames[i].c_str(), ret);
        }
    }
}

void OpenCl::setKernelArg(string kernelName, cl_uint argIndex, size_t size, void *pointer) {
    ret = clSetKernelArg(kernels[kernelName], argIndex, size, pointer);

    if (ret != CL_SUCCESS)
        fprintf(stderr, "Failed setting arg [%d] on kernel [%s]: [%d]\n", argIndex, kernelName.c_str(), ret);
}

void OpenCl::setKernelBufferArg(string kernelName, string bufferName, int argIndex) {
    ret = clSetKernelArg(kernels[kernelName], argIndex, sizeof(cl_mem), (void *)&(buffers[bufferName].buffer));
    if (ret != CL_SUCCESS)
        fprintf(stderr, "Failed setting buffer [%s] arg [%d] on kernel [%s]: %d\n", bufferName.c_str(), argIndex, kernelName.c_str(), ret);
}

void OpenCl::writeBuffer(string name, void *pointer) {
    ret = clEnqueueWriteBuffer(
        command_queue,
        buffers[name].buffer,
        CL_TRUE,
        0,
        buffers[name].size,
        pointer,
        0, NULL, NULL
    );
    
    if (ret != CL_SUCCESS) {
      fprintf(stderr, "Failed writing buffer [%s]: %d\n", name.c_str(), ret);
    }
}

void OpenCl::swapBuffers(std::string buffer1, std::string buffer2) {
   cl_mem temp = buffers[buffer1].buffer;
   buffers[buffer1].buffer = buffers[buffer2].buffer;
   buffers[buffer2].buffer = temp;
}

void OpenCl::step(string name) {
	ret = clEnqueueNDRangeKernel(command_queue, kernels[name], 2, NULL, global_item_size, NULL, 0, NULL, NULL);
    
    if (ret != CL_SUCCESS) {
      fprintf(stderr, "Failed executing kernel [%s]: %d\n", name.c_str(), ret);
    }
}

void OpenCl::readBuffer(string name, void *pointer) {
    ret = clEnqueueReadBuffer(
        command_queue,
        buffers[name].buffer,
        CL_TRUE,
        0,
        buffers[name].size,
        pointer,
        0, NULL, NULL
    );
    
    if (ret != CL_SUCCESS) {
      fprintf(stderr, "Failed reading buffer %s: %d\n", name.c_str(), ret);
    }
}

void OpenCl::cleanup() {
    map<string, cl_kernel>::iterator kernelIter;
    map<string, OpenClBuffer>::iterator bufferIter;
    
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseProgram(program);

    for (kernelIter = kernels.begin(); kernelIter != kernels.end(); kernelIter++) {
        ret = clReleaseKernel(kernelIter->second);
    }

    for (bufferIter = buffers.begin(); bufferIter != buffers.end(); bufferIter++) {
        ret = clReleaseMemObject(bufferIter->second.buffer);
    }
    
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    
    free(source_str);
}