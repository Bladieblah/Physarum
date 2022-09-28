#include <iostream>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <thread>
#include <time.h>
#include <vector>

#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <GLUT/glut.h>

#include "colour.hpp"
#include "SimplexNoise.hpp"
#include "pcg.hpp"
#include "opencl.hpp"

using namespace std;

// Window size
// #define size_x 1920
// #define size_y 1080

// #define size_x 6048
// #define size_y 3928

#define size_x 4536
#define size_y 2946

// #define size_x 3024
// #define size_y 1964

// #define size_x 5120
// #define size_y 2880

// #define size_x 2060
// #define size_y 1440

// #define size_x 1400
// #define size_y 802

int windowW = size_x / 2;
int windowH = size_y / 2;

float size_x_inv = 1. / size_x;
float size_y_inv = 1. / size_y;

bool showColorBar = false;
bool recording = false;

// Array to be drawn
uint32_t pixelData[size_y*size_x*3];

float *colourMap;
int nColours = 765;

typedef struct Particle {
    float x, y;
    float phi;
    float velocity;
} Particle;

uint32_t nParticles = 10000000;

// Idk
float sensorAngle = 45. / 180. * M_PI / 80.;
float sensorDist = 80;
float rotationAngle = 45. / 180. * M_PI / 18.;
float particleStepSize = 2;
float depositAmount = 0.05;
float stableAverage = 0.3;

// Road network! omg
// float sensorAngle = 0.4732;
// float sensorDist = 26.3819;
// float rotationAngle = 0.1338;
// float particleStepSize = 5.1793;
// float depositAmount = 0.0196;
// float stableAverage = 0.2868;

// Cloudy bu stringy??
// sensorAngle = 0.5298;
// sensorDist = 87.6185;
// rotationAngle = 2.6770;
// particleStepSize = 4.1530;
// depositAmount = 0.1068;
// stableAverage = 0.1107;

// City Grid
// float sensorAngle = 0.0474;
// float sensorDist = 14.9574;
// float rotationAngle = 0.2148;
// float particleStepSize = 2.4507;
// float depositAmount = 0.0284;
// float stableAverage = 0.1357;

// float highway
// sensorAngle = 1.0376;
// sensorDist = 40.5436;
// rotationAngle = 0.1885;
// particleStepSize = 8.7137;
// depositAmount = 0.0571;
// stableAverage = 0.1194;

// float highway with spikes
// sensorAngle = 1.3826;
// sensorDist = 40.6530;
// rotationAngle = 0.1500;
// particleStepSize = 6.9252;
// depositAmount = 0.0293;
// stableAverage = 0.1493;

// Close to clouds  
// sensorAngle = 0.1721;
// sensorDist = 108.5649;
// rotationAngle = 2.8536;
// particleStepSize = 4.3918;
// depositAmount = 0.0206;
// stableAverage = 0.2866;

// Spoopy
// float sensorAngle = 0.3596;
// float sensorDist = 158.0992;
// float rotationAngle = 0.2333;
// float particleStepSize = 6.0674;
// float depositAmount = 0.0099;
// float stableAverage = 0.3276;

// Gridsss
// sensorAngle = 0.5191;
// sensorDist = 29.3579;
// rotationAngle = 0.4350;
// particleStepSize = 8.8227;
// depositAmount = 0.2609;
// stableAverage = 0.3571;

// Ropey
// float sensorAngle = 45. / 180. * M_PI / 8.;
// float sensorDist = 20;
// float rotationAngle = 45. / 180. * M_PI / 18.;
// float particleStepSize = 2;
// float depositAmount = 0.01;
// float stableAverage = 0.2;

// Supernova
// float sensorAngle = -45. / 180. * M_PI / 80.;
// float sensorDist = 200;
// float rotationAngle = 45. / 180. * M_PI / 180.;
// float particleStepSize = 2;
// float depositAmount = 0.001;
// float stableAverage = 0.2;

// Stringy
// float sensorAngle = 45. / 180. * M_PI / 8.;
// float sensorDist = 20;
// float rotationAngle = 45. / 180. * M_PI / 18.;
// float particleStepSize = 9;
// float depositAmount = 0.01;
// float stableAverage = 0.2;

// Firey
// float sensorAngle = 45. / 180. * M_PI / 2.;
// float sensorDist = 10;
// float rotationAngle = -45. / 180. * M_PI / 6.;
// float particleStepSize = 2;
// float depositAmount = 0.1;
// float stableAverage = 0.3;

float decayFactor = 1 - (nParticles * depositAmount) / (stableAverage * size_x * size_y);
float one_9 = 1. / 9. * decayFactor;

Particle *particles;
float *trail, *trailDummy, *randomList;

// OpenCl stuff

OpenCl *opencl;
uint frameCount = 0;
uint stepCount = 0;

vector<string> bufferNames = {
    "trail",
    "trailCopy",
    "particles",
    "random",
    "image",
    "colourMap",
};

vector<size_t> bufferSizes = {
    size_x * size_y * sizeof(float),
    size_x * size_y * sizeof(float),
    nParticles * sizeof(Particle),
    (nParticles + 2) * sizeof(float),
    3 * size_x * size_y * sizeof(uint32_t),
    3 * nColours * sizeof(float),
};

vector<string> kernelNames = {
    "moveParticles",
    "diffuse",
    "depositStuff",
    "processTrail",
};

// For recording
char cmd[200];
FILE* ffmpeg;
int* buffer;

void makeColourmap() {
    std::vector<float> x_d = {0., 0.2, 0.4, 0.7, 1.};
    std::vector< std::vector<float> > y_d = {
        {26,17,36},
        {33,130,133},
        {26,17,36},
        {200,40,187},
        {241, 249, 244}
    };

    std::vector<float> x_i = {0.000, 0.032, 0.065, 0.097, 0.129, 0.161, 0.194, 0.226, 0.258, 0.290, 0.323, 0.355, 0.387, 0.419, 0.452, 0.484, 0.516, 0.548, 0.581, 0.613, 0.645, 0.677, 0.710, 0.742, 0.774, 0.806, 0.839, 0.871, 0.903, 0.935, 0.968, 1.000};
    std::vector< std::vector<float> > y_i = {
        {0,0,3},
        {3,2,18},
        {10,7,35},
        {20,11,54},
        {34,11,76},
        {48,10,92},
        {62,9,102},
        {75,12,107},
        {90,17,109},
        {102,21,110},
        {115,26,109},
        {128,31,107},
        {142,36,104},
        {155,40,100},
        {167,45,95},
        {180,51,88},
        {193,58,80},
        {204,65,72},
        {214,74,63},
        {223,84,54},
        {232,97,43},
        {239,109,33},
        {244,122,22},
        {248,136,12},
        {251,153,6},
        {251,168,13},
        {251,183,28},
        {249,199,47},
        {245,217,72},
        {241,232,100},
        {242,244,133},
        {252,254,164},
    };

    vector<float> x_w = {0., 1.};
    vector< vector<float> > y_w = {
        {0,0,0},
        {255,255,255}
    };

    Colour col(x_i, y_i, nColours);
    
    colourMap = (float *)malloc(3 * nColours * sizeof(float));
    
    col.apply(colourMap);
}

void initpixelData() {
    int i, j, k;

    // int xc = size_x / 2;
    // int yc = size_y / 2;

    // int R = size_y / 4;
    // float r, w = 5;

    // float seedx = UNI() * 20 - 10;
    // float seedy = UNI() * 20 - 10;

    int ind;

    for (i = 0; i < size_x; i++) {
        for (j = 0; j < size_y; j++) {
            ind = i + size_x * j;
            // r = sqrt(pow(i - xc, 2) + pow(j - yc, 2));
            // trail[ind] = fmin(1., exp(-pow((r - R) / w, 2)));// + pow((1 + SimplexNoise::noise(i * size_y_inv * 60, j * size_y_inv * 60)) / 2, 6) / 2);
            // trail[i][j] = fmin(1., pow((1 + SimplexNoise::noise((i * size_y_inv + seedx) * 10, (j * size_y_inv + seedy) * 10)) / 2, 6) / 2);
            trail[ind] = 0;
            for (k = 0; k < 3; k++) {
                pixelData[3 * ind + k] = 0;
            }
        }
    }
}

float clip(float in, float lower, float upper) {
    if (in < lower) {
        return lower;
    }
    if (in > upper) {
        return upper;
    }

    return in;
}

void initParticles() {
    int i;

    // Squaretangle
    // for (i = 0; i < particlesPerThread; i++) {
    //     particle = particles[thread][i];

    //     particle.x = clip(0.5 + RANDN() * 0.31999, 0.3, 0.7) * size_x;
    //     particle.y = clip(0.5 + RANDN() * 0.31999, 0.3, 0.7) * size_y;
    //     particle.phi = 2 * M_PI * UNI();

    //     particles[thread][i] = particle;
    // }

    // Circle
    float xc = size_x * 0.5;
    float yc = size_y * 0.5;

    for (i = 0; i < nParticles; i++) {
        float theta = UNI() * 2 * M_PI;
        float rad = (RANDN() / 32. + 0.25);

        particles[i].x = clip(cos(theta) * rad * size_y + xc, 0., size_x);
        particles[i].y = clip(sin(theta) * rad * size_y + yc, 0., size_y);
        particles[i].phi = atan2(yc - particles[i].y, xc - particles[i].x);
        particles[i].velocity = UNI() * particleStepSize + 1;
    }
}

void fillRandom() {
    for (int i = 0; i < nParticles + 2; i++) {
        randomList[i] = UNI();
    }
}

void writeBuffers() {
    opencl->writeBuffer("trail", (void *)trail);
    opencl->writeBuffer("trailCopy", (void *)trail);
    opencl->writeBuffer("particles", (void *)particles);
    opencl->writeBuffer("random", (void *)randomList);
    opencl->writeBuffer("image", (void *)pixelData);
    opencl->writeBuffer("colourMap", (void *)colourMap);
}

void prepare() {
    opencl = new OpenCl(
        size_x,
        size_y,
        "shaders/physarum.cl",
        true,
        bufferNames,
        bufferSizes,
        kernelNames
    );

    pcg32_srandom(time(NULL) ^ (intptr_t)&printf, (intptr_t)&nParticles); // Seed pcg

    particles = (Particle *)malloc(nParticles * sizeof(Particle));
    trail = (float *)malloc(size_x * size_y * sizeof(float));
    trailDummy = (float *)malloc(size_x * size_y * sizeof(float));
    randomList = (float *)malloc((nParticles + 2) * sizeof(float));

    initpixelData();

    for (int i = 0; i < nParticles; i++) {
        particles[i] = Particle();
    }

    initParticles();
    fillRandom();
    makeColourmap();

    writeBuffers();

    // Diffuse with dual
    opencl->setKernelBufferArg("diffuse", "trail", 0);
    opencl->setKernelBufferArg("diffuse", "trailCopy", 1);
    opencl->setKernelArg("diffuse", 2, sizeof(float), (void *)&one_9);

    opencl->setKernelBufferArg("diffuse2", "trailCopy", 0);
    opencl->setKernelBufferArg("diffuse2", "trail", 1);
    opencl->setKernelArg("diffuse2", 2, sizeof(float), (void *)&one_9);

    opencl->setKernelBufferArg("moveParticles", "particles", 0);
    opencl->setKernelBufferArg("moveParticles", "trail", 1);
    opencl->setKernelBufferArg("moveParticles", "random", 2);

    opencl->setKernelBufferArg("moveParticles2", "particles", 0);
    opencl->setKernelBufferArg("moveParticles2", "trailCopy", 1);
    opencl->setKernelBufferArg("moveParticles2", "random", 2);

    int size_x2 = size_x, size_y2 = size_y;
    opencl->setKernelArg("moveParticles", 3, sizeof(int), (void *)&size_x2);
    opencl->setKernelArg("moveParticles", 4, sizeof(int), (void *)&size_y2);
    opencl->setKernelArg("moveParticles", 5, sizeof(float), (void *)&sensorAngle);
    opencl->setKernelArg("moveParticles", 6, sizeof(float), (void *)&sensorDist);
    opencl->setKernelArg("moveParticles", 7, sizeof(float), (void *)&rotationAngle);

    opencl->setKernelArg("moveParticles2", 3, sizeof(int), (void *)&size_x2);
    opencl->setKernelArg("moveParticles2", 4, sizeof(int), (void *)&size_y2);
    opencl->setKernelArg("moveParticles2", 5, sizeof(float), (void *)&sensorAngle);
    opencl->setKernelArg("moveParticles2", 6, sizeof(float), (void *)&sensorDist);
    opencl->setKernelArg("moveParticles2", 7, sizeof(float), (void *)&rotationAngle);

    opencl->setKernelBufferArg("depositStuff", "particles", 0);
    opencl->setKernelBufferArg("depositStuff", "trail", 1);
    opencl->setKernelArg("depositStuff", 2, sizeof(int), (void *)&size_x2);
    opencl->setKernelArg("depositStuff", 3, sizeof(int), (void *)&size_y2);
    opencl->setKernelArg("depositStuff", 4, sizeof(float), (void *)&depositAmount);

    opencl->setKernelBufferArg("depositStuff2", "particles", 0);
    opencl->setKernelBufferArg("depositStuff2", "trailCopy", 1);
    opencl->setKernelArg("depositStuff2", 2, sizeof(int), (void *)&size_x2);
    opencl->setKernelArg("depositStuff2", 3, sizeof(int), (void *)&size_y2);
    opencl->setKernelArg("depositStuff2", 4, sizeof(float), (void *)&depositAmount);

    opencl->setKernelBufferArg("processTrail", "trail", 0);
    opencl->setKernelBufferArg("processTrail", "image", 1);
    opencl->setKernelBufferArg("processTrail", "colourMap", 2);
    opencl->setKernelArg("processTrail", 3, sizeof(int), (void *)&nColours);

    opencl->setKernelBufferArg("processTrail2", "trailCopy", 0);
    opencl->setKernelBufferArg("processTrail2", "image", 1);
    opencl->setKernelBufferArg("processTrail2", "colourMap", 2);
    opencl->setKernelArg("processTrail2", 3, sizeof(int), (void *)&nColours);
}

void cleanup() {
    /* Finalization */
    free(colourMap);
    free(particles);
    free(trail);
    free(trailDummy);
    free(randomList);

    opencl->cleanup();
}

void moveParticles() {
    size_t particle_item_size[1] = {nParticles};

    if (stepCount % 2 == 0) {
        opencl->ret = clEnqueueNDRangeKernel(
            opencl->command_queue, 
            opencl->kernels["moveParticles"], 
            1, NULL, 
            particle_item_size, 
            NULL, 0, NULL, NULL
        );
    } else {
        opencl->ret = clEnqueueNDRangeKernel(
            opencl->command_queue, 
            opencl->kernels["moveParticles2"], 
            1, NULL, 
            particle_item_size, 
            NULL, 0, NULL, NULL
        );
    }
    
    if (opencl->ret != CL_SUCCESS) {
      fprintf(stderr, "Failed executing kernel [moveParticles]: %d\n", opencl->ret);
    }

    fillRandom();
    opencl->writeBuffer("random", (void *)randomList);
}

void depositStuff() {
    size_t particle_item_size[1] = {nParticles};

    if (stepCount % 2 == 0) {
        opencl->ret = clEnqueueNDRangeKernel(
            opencl->command_queue, 
            opencl->kernels["depositStuff"], 
            1, NULL, 
            particle_item_size, 
            NULL, 0, NULL, NULL
        );
    } else {
        opencl->ret = clEnqueueNDRangeKernel(
            opencl->command_queue, 
            opencl->kernels["depositStuff2"], 
            1, NULL, 
            particle_item_size, 
            NULL, 0, NULL, NULL
        );
    }
    
    if (opencl->ret != CL_SUCCESS) {
      fprintf(stderr, "Failed executing kernel [depositStuff]: %d\n", opencl->ret);
    }
}

void iterParticles() {
	moveParticles();
	depositStuff();
}

void diffuse() {
    if (stepCount % 2 == 0) {
        opencl->step("diffuse");
    } else {
        opencl->step("diffuse2");
    }
}

void calculateImage() {
    if (stepCount % 2 == 0) {
        opencl->step("processTrail");
    } else {
        opencl->step("processTrail2");
    }

    opencl->readBuffer("image", (void *)&pixelData[0]);
    opencl->readBuffer("trail", (void *)trail);
}

void step() {
    iterParticles();
    diffuse();
    calculateImage();
    stepCount++;
}

void display() {
    if (frameCount % 2 == 0) {
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

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
            &pixelData[0]
        );

        glBegin(GL_QUADS);
            glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0, -1.0);
            glTexCoord2f(1.0f, 0.0f); glVertex2f( 1.0, -1.0);
            glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.0,  1.0);
            glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0,  1.0);
        glEnd();

<<<<<<< HEAD
        glFlush();
=======
        // glFlush();
>>>>>>> da56631317e62512ba791519844d92ce30277874
        glutSwapBuffers();

        if (recording) {
            glReadPixels(0, 0, 1512, 916, GL_RGBA, GL_UNSIGNED_BYTE, buffer);
            fwrite(buffer, sizeof(int)*1512*916, 1, ffmpeg);
        }
        
        step();
        
<<<<<<< HEAD
        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> time_span = std::chrono::duration_cast<std::chrono::duration<float>>(t2 - t1);
        fprintf(stderr, "\rStep = %d, time = %.4g            ", frameCount, time_span.count());
    }

=======

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> time_span = std::chrono::duration_cast<std::chrono::duration<float>>(t2 - t1);
        fprintf(stderr, "\rStep = %d, time = %.4g            ", frameCount / 2, time_span.count());

    }

>>>>>>> da56631317e62512ba791519844d92ce30277874
    frameCount++;
}

void randomiseParameters() {
    sensorAngle = 2 * UNI() * M_PI;
    sensorDist = UNI() * 200;
    rotationAngle = 2 * UNI() * M_PI;
    particleStepSize = UNI() * 10;
    depositAmount = exp(UNI() * 4 - 5.5);
    stableAverage = UNI() * 0.3 + 0.1;

    decayFactor = 1 - (nParticles * depositAmount) / (stableAverage * size_x * size_y);
    one_9 = 1. / 9. * decayFactor;

    fprintf(stderr, "\nsensorAngle = %.4f;\nsensorDist = %.4f;\nrotationAngle = %.4f;\nparticleStepSize = %.4f;\ndepositAmount = %.4f;\nstableAverage = %.4f;\n\n",
        sensorAngle, sensorDist, rotationAngle, particleStepSize, depositAmount, stableAverage);

    opencl->setKernelArg("diffuse", 2, sizeof(float), (void *)&one_9);
    opencl->setKernelArg("moveParticles", 5, sizeof(float), (void *)&sensorAngle);
    opencl->setKernelArg("moveParticles", 6, sizeof(float), (void *)&sensorDist);
    opencl->setKernelArg("moveParticles", 7, sizeof(float), (void *)&rotationAngle);
    opencl->setKernelArg("depositStuff", 4, sizeof(float), (void *)&depositAmount);

    opencl->setKernelArg("diffuse2", 2, sizeof(float), (void *)&one_9);
    opencl->setKernelArg("moveParticles2", 5, sizeof(float), (void *)&sensorAngle);
    opencl->setKernelArg("moveParticles2", 6, sizeof(float), (void *)&sensorDist);
    opencl->setKernelArg("moveParticles2", 7, sizeof(float), (void *)&rotationAngle);
    opencl->setKernelArg("depositStuff2", 4, sizeof(float), (void *)&depositAmount);
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
            initpixelData();
            break;
        case 'i':
            initParticles();
            break;
        case 'b':
            showColorBar = !showColorBar;
            break;
        case 'u':
            randomiseParameters();
            break;
        case 'q':
        	cleanup();
        	fprintf(stderr, "\n");

            if (recording)
                pclose(ffmpeg);
            
            exit(0);
            break;
        default:
            break;
    }
}

void reshape(int w, int h)
{
    windowW = w;
    windowH = h;

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

    // if (recording)
    glViewport(0, 0, w, h);
    
	glMatrixMode(GL_MODELVIEW);

    fprintf(stderr, "w, h = %d, %d\n", w, h);
}

void setupRecording() {
    // start ffmpeg telling it to expect raw rgba 720p-60hz frames
    // -i - tells it to read frames from stdin
    sprintf(cmd, "ffmpeg -r 60 -f rawvideo -pix_fmt rgba -s %dx%d -i - -threads 0 -preset fast -y -pix_fmt yuv420p -crf 21 -vf vflip output.mp4", 1512, 916);
    // open pipe to ffmpeg's stdin in binary write mode
    ffmpeg = popen(cmd, "w");
    buffer = new int[1512*916];
}

int main(int argc, char **argv) {
    if (argc > 1) {
        if (strcmp(argv[1], "-s") == 0) {
            recording = true;
        }
    }

    prepare();

    if (recording)
        setupRecording();
    
	glutInit( &argc, argv );
    glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE );
    glutInitWindowSize( size_x, size_y );
    glutCreateWindow( "Physarum" );
    
    glutDisplayFunc(&display);
    glutIdleFunc(&display);
    glutKeyboardUpFunc(&key_pressed);
    glutReshapeFunc(&reshape);
    
    glutMainLoop();

    return 0;
}	