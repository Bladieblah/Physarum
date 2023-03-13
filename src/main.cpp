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

// #define size_x 4536
// #define size_y 2946

#define size_x 3024
#define size_y 1964

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
bool renderTrail = false;

// Array to be drawn
uint32_t pixelData[size_y*size_x*3];

float *colourMap;
int nColours = 765;

typedef struct Particle {
    float x, y;
    float phi;
    float velocity;
} Particle;

uint32_t nParticles = 4000000;

// Idk
// float sensorAngle = 45. / 180. * M_PI / 80.;
// float sensorDist = 80;
// float rotationAngle = 45. / 180. * M_PI / 18.;
// float particleStepSize = 2;
// float depositAmount = 0.05;
// float stableAverage = 0.3;

// Road network! omg
// float sensorAngle = 0.4732;
// float sensorDist = 26.3819;
// float rotationAngle = 0.1338;
// float particleStepSize = 5.1793;
// float depositAmount = 0.0196;
// float stableAverage = 0.2868;

// Cloudy bu stringy??
// float sensorAngle = 0.5298;
// float sensorDist = 87.6185;
// float rotationAngle = 2.6770;
// float particleStepSize = 4.1530;
// float depositAmount = 0.1068;
// float stableAverage = 0.1107;

// City Grid
// float sensorAngle = 0.0474;
// float sensorDist = 14.9574;
// float rotationAngle = 0.2148;
// float particleStepSize = 2.4507;
// float depositAmount = 0.0284;
// float stableAverage = 0.1357;

// double highway
// float sensorAngle = 1.0376;
// float sensorDist = 40.5436;
// float rotationAngle = 0.1885;
// float particleStepSize = 8.7137;
// float depositAmount = 0.0571;
// float stableAverage = 0.1194;

// double highway with spikes
// float sensorAngle = 1.3826;
// float sensorDist = 40.6530;
// float rotationAngle = 0.1500;
// float particleStepSize = 6.9252;
// float depositAmount = 0.0293;
// float stableAverage = 0.1493;

// Highway 2
// float sensorAngle = 5.7312;
// float sensorDist = 104.3958;
// float rotationAngle = 5.7027;
// float particleStepSize = 0.3103;
// float depositAmount = 0.0625;
// float stableAverage = 0.2133;

// Close to clouds  
// float sensorAngle = 0.1721;
// float sensorDist = 108.5649;
// float rotationAngle = 2.8536;
// float particleStepSize = 4.3918;
// float depositAmount = 0.0206;
// float stableAverage = 0.2866;

// Spoopy
// float sensorAngle = 0.3596;
// float sensorDist = 158.0992;
// float rotationAngle = 0.2333;
// float particleStepSize = 6.0674;
// float depositAmount = 0.0099;
// float stableAverage = 0.3276;

// Gridsss
// float sensorAngle = 0.5191;
// float sensorDist = 29.3579;
// float rotationAngle = 0.4350;
// float particleStepSize = 8.8227;
// float depositAmount = 0.2609;
// float stableAverage = 0.3571;

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

// semi-cloudy
// float sensorAngle = 5.1340;
// float sensorDist = 8.6095;
// float rotationAngle = 2.4484;
// float particleStepSize = 7.8162;
// float depositAmount = 0.0140;
// float stableAverage = 0.2407;

// Sauron
// float sensorAngle = 4.4548;
// float sensorDist = 45.7309;
// float rotationAngle = 6.1194;
// float particleStepSize = 7.2887;
// float depositAmount = 0.1714;
// float stableAverage = 0.3403;

// Honingraat
// float sensorAngle = 2.2544;
// float sensorDist = 4.7555;
// float rotationAngle = 4.9923;
// float particleStepSize = 9.2328;
// float depositAmount = 0.2203;
// float stableAverage = 0.3875;

// Very dynamic
// float sensorAngle = 5.9553;
// float sensorDist = 197.3020;
// float rotationAngle = 6.2584;
// float particleStepSize = 6.8858;
// float depositAmount = 0.0141;
// float stableAverage = 0.2476;

// Keeps evolving
// float sensorAngle = 0.1088;
// float sensorDist = 179.6908;
// float rotationAngle = 0.1832;
// float particleStepSize = 1.5837;
// float depositAmount = 0.0781;
// float stableAverage = 0.3475;

float sensorAngle = 0.4367;
float sensorDist = 13.2189;
float rotationAngle = 0.7218;
float particleStepSize = 5.9052;
float depositAmount = 0.1264;
float stableAverage = 0.2992;

float decayFactor = 1 - (nParticles * depositAmount) / (stableAverage * size_x * size_y);
float one_9 = 1. / 9. * decayFactor;

Particle *particles;
float *trail, *trailDummy;

// OpenCl stuff

OpenCl *opencl;
uint frameCount = 0;
uint stepCount = 0;

vector<BufferSpec> bufferSpecs;
void createBufferSpecs() {
    bufferSpecs = {
        {"trail",     {NULL, size_x * size_y * sizeof(float)}},
        {"trailCopy", {NULL, size_x * size_y * sizeof(float)}},
        {"particles", {NULL, nParticles * sizeof(Particle)}},
        {"random",    {NULL, (nParticles + 2) * sizeof(float)}},
        {"image",     {NULL, 3 * size_x * size_y * sizeof(uint32_t)}},
        {"image2",    {NULL, 3 * size_x * size_y * sizeof(uint32_t)}},
        {"colourMap", {NULL, 3 * nColours * sizeof(float)}},

        {"randomState",     {NULL, nParticles * sizeof(uint64_t)}},
        {"randomIncrement", {NULL, nParticles * sizeof(uint64_t)}},
        {"initState",       {NULL, nParticles * sizeof(uint64_t)}},
        {"initSeq",         {NULL, nParticles * sizeof(uint64_t)}},
    };
}

vector<KernelSpec> kernelSpecs;
void createKernelSpecs() {
    kernelSpecs = {
        {"seedNoise",       {NULL, 1, {nParticles, 0},  {0, 0}, "seedNoise"}},
        {"moveParticles1",  {NULL, 1, {nParticles, 0},  {0, 0}, "moveParticles"}},
        {"moveParticles2",  {NULL, 1, {nParticles, 0},  {0, 0}, "moveParticles"}},
        {"depositStuff1",   {NULL, 1, {nParticles, 0},  {0, 0}, "depositStuff"}},
        {"depositStuff2",   {NULL, 1, {nParticles, 0},  {0, 0}, "depositStuff"}},
        {"renderParticles", {NULL, 1, {nParticles, 0},  {0, 0}, "renderParticles"}},
        {"diffuse1",        {NULL, 2, {size_x, size_y}, {0, 0}, "diffuse"}},
        {"diffuse2",        {NULL, 2, {size_x, size_y}, {0, 0}, "diffuse"}},
        {"processTrail1",   {NULL, 2, {size_x, size_y}, {0, 0}, "processTrail"}},
        {"processTrail2",   {NULL, 2, {size_x, size_y}, {0, 0}, "processTrail"}},
        {"resetImage",      {NULL, 2, {size_x, size_y}, {0, 0}, "resetImage"}},
        {"invertImage",     {NULL, 2, {size_x, size_y}, {0, 0}, "invertImage"}},
        {"lagImage",        {NULL, 2, {size_x, size_y}, {0, 0}, "lagImage"}},
    };
}

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

    opencl->writeBuffer("colourMap", (void *)colourMap);
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

    opencl->writeBuffer("trail", (void *)trail);
    opencl->writeBuffer("trailCopy", (void *)trail);
    opencl->writeBuffer("image", (void *)pixelData);
    opencl->writeBuffer("image2", (void *)pixelData);
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
    
    opencl->writeBuffer("particles", (void *)particles);
}

void setKernelArgs() {
    int size_x2 = size_x, size_y2 = size_y;

    opencl->setKernelBufferArg("seedNoise", 0, "randomState");
    opencl->setKernelBufferArg("seedNoise", 1, "randomIncrement");
    opencl->setKernelBufferArg("seedNoise", 2, "initState");
    opencl->setKernelBufferArg("seedNoise", 3, "initSeq");

    opencl->setKernelBufferArg("diffuse1", 0, "trail");
    opencl->setKernelBufferArg("diffuse1", 1, "trailCopy");
    opencl->setKernelArg("diffuse1", 2, sizeof(float), (void *)&one_9);

    opencl->setKernelBufferArg("diffuse2", 0, "trailCopy");
    opencl->setKernelBufferArg("diffuse2", 1, "trail");
    opencl->setKernelArg("diffuse2", 2, sizeof(float), (void *)&one_9);

    opencl->setKernelBufferArg("moveParticles1", 0, "particles");
    opencl->setKernelBufferArg("moveParticles1", 1, "trail");
    opencl->setKernelBufferArg("moveParticles1", 2, "randomState");
    opencl->setKernelBufferArg("moveParticles1", 3, "randomIncrement");
    opencl->setKernelArg("moveParticles1", 4, sizeof(int), (void *)&size_x2);
    opencl->setKernelArg("moveParticles1", 5, sizeof(int), (void *)&size_y2);
    opencl->setKernelArg("moveParticles1", 6, sizeof(float), (void *)&sensorAngle);
    opencl->setKernelArg("moveParticles1", 7, sizeof(float), (void *)&sensorDist);
    opencl->setKernelArg("moveParticles1", 8, sizeof(float), (void *)&rotationAngle);

    opencl->setKernelBufferArg("moveParticles2", 0, "particles");
    opencl->setKernelBufferArg("moveParticles2", 1, "trailCopy");
    opencl->setKernelBufferArg("moveParticles2", 2, "randomState");
    opencl->setKernelBufferArg("moveParticles2", 3, "randomIncrement");
    opencl->setKernelArg("moveParticles2", 4, sizeof(int), (void *)&size_x2);
    opencl->setKernelArg("moveParticles2", 5, sizeof(int), (void *)&size_y2);
    opencl->setKernelArg("moveParticles2", 6, sizeof(float), (void *)&sensorAngle);
    opencl->setKernelArg("moveParticles2", 7, sizeof(float), (void *)&sensorDist);
    opencl->setKernelArg("moveParticles2", 8, sizeof(float), (void *)&rotationAngle);

    opencl->setKernelBufferArg("depositStuff1", 0, "particles");
    opencl->setKernelBufferArg("depositStuff1", 1, "trail");
    opencl->setKernelArg("depositStuff1", 2, sizeof(int), (void *)&size_x2);
    opencl->setKernelArg("depositStuff1", 3, sizeof(int), (void *)&size_y2);
    opencl->setKernelArg("depositStuff1", 4, sizeof(float), (void *)&depositAmount);

    opencl->setKernelBufferArg("depositStuff2", 0, "particles");
    opencl->setKernelBufferArg("depositStuff2", 1, "trailCopy");
    opencl->setKernelArg("depositStuff2", 2, sizeof(int), (void *)&size_x2);
    opencl->setKernelArg("depositStuff2", 3, sizeof(int), (void *)&size_y2);
    opencl->setKernelArg("depositStuff2", 4, sizeof(float), (void *)&depositAmount);

    opencl->setKernelBufferArg("processTrail1", 0, "trail");
    opencl->setKernelBufferArg("processTrail1", 1, "image");
    opencl->setKernelBufferArg("processTrail1", 2, "colourMap");
    opencl->setKernelArg("processTrail1", 3, sizeof(int), (void *)&nColours);

    opencl->setKernelBufferArg("processTrail2", 0, "trailCopy");
    opencl->setKernelBufferArg("processTrail2", 1, "image");
    opencl->setKernelBufferArg("processTrail2", 2, "colourMap");
    opencl->setKernelArg("processTrail2", 3, sizeof(int), (void *)&nColours);

    opencl->setKernelBufferArg("resetImage", 0, "image");

    opencl->setKernelBufferArg("renderParticles", 0, "particles");
    opencl->setKernelBufferArg("renderParticles", 1, "image");
    opencl->setKernelArg("renderParticles", 2, sizeof(int), &size_x2);

    opencl->setKernelBufferArg("invertImage", 0, "image");

    opencl->setKernelBufferArg("lagImage", 0, "image");
    opencl->setKernelBufferArg("lagImage", 1, "image2");
}

void initPcg() {
    uint64_t *initState, *initSeq;
    initState = (uint64_t *)malloc(nParticles * sizeof(uint64_t));
    initSeq = (uint64_t *)malloc(nParticles * sizeof(uint64_t));

    for (int i = 0; i < nParticles; i++) {
        initState[i] = pcg32_random();
        initSeq[i] = pcg32_random();
    }

    opencl->writeBuffer("initState", (void *)initState);
    opencl->writeBuffer("initSeq", (void *)initSeq);
    opencl->step("seedNoise");
    opencl->flush();

    free(initState);
    free(initSeq);
}

void prepare() {
    createBufferSpecs();
    createKernelSpecs();

    opencl = new OpenCl(
        "shaders/physarum.cl",
        bufferSpecs,
        kernelSpecs
    );

    pcg32_srandom(time(NULL) ^ (intptr_t)&printf, (intptr_t)&nParticles); // Seed pcg

    particles = (Particle *)malloc(nParticles * sizeof(Particle));
    trail = (float *)malloc(size_x * size_y * sizeof(float));
    trailDummy = (float *)malloc(size_x * size_y * sizeof(float));

    for (int i = 0; i < nParticles; i++) {
        particles[i] = Particle();
    }

    setKernelArgs();

    initPcg();
    initpixelData();
    initParticles();
    makeColourmap();
}

void cleanup() {
    /* Finalization */
    free(colourMap);
    free(particles);
    free(trail);
    free(trailDummy);

    opencl->cleanup();
}

void moveParticles() {
    if (stepCount % 2 == 0) {
        opencl->step("moveParticles1");
    } else {
        opencl->step("moveParticles2");
    }
}

void depositStuff() {
    if (stepCount % 2 == 0) {
        opencl->step("depositStuff1");
    } else {
        opencl->step("depositStuff2");
    }
}

void iterParticles() {
	moveParticles();
	depositStuff();
}

void diffuse() {
    if (stepCount % 2 == 0) {
        opencl->step("diffuse1");
    } else {
        opencl->step("diffuse2");
    }
}

void calculateImage() {
    if (renderTrail) {
        if (stepCount % 2 == 0) {
            opencl->step("processTrail1");
        } else {
            opencl->step("processTrail2");
        }
        opencl->readBuffer("image", (void *)&pixelData[0]);
    } else {
        opencl->step("resetImage");
        opencl->step("renderParticles");
        opencl->step("invertImage");
        opencl->step("lagImage");
        opencl->readBuffer("image2", (void *)&pixelData[0]);
    }
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

        glFlush();
        glutSwapBuffers();

        if (recording) {
            glReadPixels(0, 0, 1512, 916, GL_RGBA, GL_UNSIGNED_BYTE, buffer);
            fwrite(buffer, sizeof(int)*1512*916, 1, ffmpeg);
        }
        
        step();

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> time_span = std::chrono::duration_cast<std::chrono::duration<float>>(t2 - t1);
        fprintf(stderr, "Step = %d, time = %.4g            ", frameCount / 2, time_span.count());
        if (renderTrail) {
            fprintf(stderr, "\x1b[4A");
        } else {
            fprintf(stderr, "\x1b[7A");
        }

    }

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
        case 't':
            renderTrail = !renderTrail;
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