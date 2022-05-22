#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>
#include <math.h>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <GLUT/glut.h>

#include "colour.hpp"

// For loading shader files
#define MAX_SOURCE_SIZE (0x100000)

// Window size
#define size_x 1920
#define size_y 1080

// OpenCL initialisation
cl_platform_id platform_id = NULL;
cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue command_queue = NULL;

cl_mem mapmobj = NULL;
cl_mem datamobj = NULL;

cl_program program = NULL;
cl_kernel kernel = NULL;
cl_uint ret_num_devices;
cl_uint ret_num_platforms;
cl_int ret;

size_t source_size;
char *source_str;

char to_change;

// Array to be drawn
unsigned int data[size_y*size_x*3];

float *colourMap;
int nColours = 255;

// Kernel size for parallelisation
size_t global_item_size[2] = {(size_t)size_x, (size_t)size_y};
size_t local_item_size[2] = {(size_t)size_x, (size_t)size_y};

void makeColourmap() {
    std::vector<float> x = {0., 0.2, 0.4, 0.7, 1.};
    std::vector< std::vector<float> > y = {
        {26,17,36},
        {33,130,133},
        {26,17,36},
        {200,40,187},
        {241, 249, 244}
    };

    Colour col(x, y, nColours);
    
    colourMap = (float *)malloc(3 * nColours * sizeof(float));
    
    col.apply(colourMap);
    
    // Write colourmap to GPU
    ret = clEnqueueWriteBuffer(command_queue, mapmobj, CL_TRUE, 0, 3*nColours*sizeof(float), colourMap, 0, NULL, NULL);
}

void setKernelArgs() {
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&mapmobj);
    ret = clSetKernelArg(kernel, 1, sizeof(int), &nColours);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&datamobj);
}

void initData() {
    return;
}

void prepare() {
    FILE *fp;
    const char fileName[] = "./shaders/sample.cl";
    
    fp = fopen(fileName, "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char *)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);
    
    srand(time(NULL));
    
    initData();
    
    /* Get Platform/Device Information */
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);

    /* Create OpenCL Context */
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

    /* Create command queue */
    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    /* Create Buffer Object */
    mapmobj = clCreateBuffer(context, CL_MEM_READ_WRITE, 3*nColours*sizeof(float), NULL, &ret);
    datamobj = clCreateBuffer(context, CL_MEM_READ_WRITE, 3*size_x*size_y*sizeof(unsigned int), NULL, &ret);

    /* Create kernel program from source file*/
    program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    
    size_t len = 10000;
    ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
    char *buffer = (char *)calloc(len, sizeof(char));
    ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
    fprintf(stderr, "%s\n", buffer);

    /* Create data parallel OpenCL kernel */
    kernel = clCreateKernel(program, "rdKernel", &ret);
    setKernelArgs();
}

void cleanup() {
    /* Finalization */
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    
    ret = clReleaseMemObject(mapmobj);
    ret = clReleaseMemObject(datamobj);
    
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

    free(colourMap);
    
    free(source_str);
}

void step() {
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_item_size, NULL, 0, NULL, NULL);
    ret = clEnqueueReadBuffer(command_queue, datamobj, CL_TRUE, 0, 3*size_x*size_y*sizeof(unsigned int), data, 0, NULL, NULL);
}

void display() {
    glClearColor( 0, 0, 0, 1 );
    glClear( GL_COLOR_BUFFER_BIT );

    glEnable (GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    glTexImage2D (
        GL_TEXTURE_2D,
        0,
        GL_RGB,
        size_x,
        size_y,
        0,
        GL_RGB,
        GL_UNSIGNED_INT,
        &data[0]
    );

    glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0, -1.0);
        glTexCoord2f(1.0f, 0.0f); glVertex2f( 1.0, -1.0);
        glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.0,  1.0);
        glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0,  1.0);
    glEnd();

    glFlush();
    glutSwapBuffers();
    
    step();
}

void key_pressed(unsigned char key, int x, int y) {
    switch (key)
    {
        case 'p':
            glutIdleFunc(&display);
            break;
        case 'e':
            glutPostRedisplay();
            break;
        case 'r':
            initData();
            break;
        case 'q':
        	cleanup();
        	fprintf(stderr, "\n");
            exit(0);
            break;
        default:
            break;
    }
}

int main(int argc, char **argv) {
    prepare();
    makeColourmap();
    
	glutInit( &argc, argv );
    glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE );
    glutInitWindowSize( size_x, size_y );
    glutCreateWindow( "Hello World" );
    glutDisplayFunc( display );
    
    glutDisplayFunc(&display);
//     glutIdleFunc(&display);
    glutKeyboardUpFunc(&key_pressed);
    
    glutMainLoop();

    return 0;
}	