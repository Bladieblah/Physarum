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

#include "colourMap.hpp"
#include "config.hpp"
#include "SimplexNoise.hpp"
#include "pcg.hpp"
#include "opencl.hpp"
#include "mainWindow.hpp"

using namespace std;

int windowW, windowH;

float size_x_inv, size_y_inv;
bool recording = false;

chrono::high_resolution_clock::time_point timePoint;
unsigned int frameCount = 0;
float frameTime = 0;
uint32_t iterCount = 0;
uint32_t stepCount = 0;

unsigned int *cmap;
ColourMap *cm;

typedef struct Particle {
    float x, y;
    float phi;
    float velocity;
} Particle;


// OpenCl stuff

OpenCl *opencl;
Config *config;

vector<BufferSpec> bufferSpecs;
void createBufferSpecs() {
    bufferSpecs = {
        {"trail",     {NULL, config->width * config->height * sizeof(float)}},
        {"trailCopy", {NULL, config->width * config->height * sizeof(float)}},
        {"particles", {NULL, config->particleCount * sizeof(Particle)}},
        {"random",    {NULL, (config->particleCount + 2) * sizeof(float)}},
        {"image",     {NULL, 3 * config->width * config->height * sizeof(uint32_t)}},
        {"image2",    {NULL, 3 * config->width * config->height * sizeof(uint32_t)}},
        {"colourMap", {NULL, 3 * config->num_colours * sizeof(unsigned int)}},

        {"randomState",     {NULL, config->particleCount * sizeof(uint64_t)}},
        {"randomIncrement", {NULL, config->particleCount * sizeof(uint64_t)}},
        {"initState",       {NULL, config->particleCount * sizeof(uint64_t)}},
        {"initSeq",         {NULL, config->particleCount * sizeof(uint64_t)}},
    };
}

vector<KernelSpec> kernelSpecs;
void createKernelSpecs() {
    kernelSpecs = {
        {"seedNoise",       {NULL, 1, {config->particleCount, 0},  {0, 0}, "seedNoise"}},
        {"initParticles",   {NULL, 1, {config->particleCount, 0},  {0, 0}, "initParticles"}},
        {"setParticleVels", {NULL, 1, {config->particleCount, 0},  {0, 0}, "setParticleVels"}},
        {"moveParticles1",  {NULL, 1, {config->particleCount, 0},  {0, 0}, "moveParticles"}},
        {"moveParticles2",  {NULL, 1, {config->particleCount, 0},  {0, 0}, "moveParticles"}},
        {"depositStuff1",   {NULL, 1, {config->particleCount, 0},  {0, 0}, "depositStuff"}},
        {"depositStuff2",   {NULL, 1, {config->particleCount, 0},  {0, 0}, "depositStuff"}},
        {"renderParticles", {NULL, 1, {config->particleCount, 0},  {0, 0}, "renderParticles"}},
        {"diffuse1",        {NULL, 2, {config->width, config->height}, {0, 0}, "diffuse"}},
        {"diffuse2",        {NULL, 2, {config->width, config->height}, {0, 0}, "diffuse"}},
        {"resetTrail",      {NULL, 2, {config->width, config->height}, {0, 0}, "resetTrail"}},
        {"processTrail1",   {NULL, 2, {config->width, config->height}, {0, 0}, "processTrail"}},
        {"processTrail2",   {NULL, 2, {config->width, config->height}, {0, 0}, "processTrail"}},
        {"resetImage",      {NULL, 2, {config->width, config->height}, {0, 0}, "resetImage"}},
        {"invertImage",     {NULL, 2, {config->width, config->height}, {0, 0}, "invertImage"}},
        {"lagImage",        {NULL, 2, {config->width, config->height}, {0, 0}, "lagImage"}},
    };
}

// For recording
char cmd[200];
FILE* ffmpeg;
int* buffer;

void setKernelArgs() {
    float decayFactor = 1 - (config->particleCount * config->depositAmount) / (config->stableAverage * config->width * config->height);
    float one_9 = 1. / 9. * decayFactor;

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
    opencl->setKernelArg("moveParticles1", 4, sizeof(int), (void *)&(config->width));
    opencl->setKernelArg("moveParticles1", 5, sizeof(int), (void *)&(config->height));
    opencl->setKernelArg("moveParticles1", 6, sizeof(float), (void *)&(config->sensorAngle));
    opencl->setKernelArg("moveParticles1", 7, sizeof(float), (void *)&(config->sensorDist));
    opencl->setKernelArg("moveParticles1", 8, sizeof(float), (void *)&(config->rotationAngle));

    opencl->setKernelBufferArg("moveParticles2", 0, "particles");
    opencl->setKernelBufferArg("moveParticles2", 1, "trailCopy");
    opencl->setKernelBufferArg("moveParticles2", 2, "randomState");
    opencl->setKernelBufferArg("moveParticles2", 3, "randomIncrement");
    opencl->setKernelArg("moveParticles2", 4, sizeof(int), (void *)&(config->width));
    opencl->setKernelArg("moveParticles2", 5, sizeof(int), (void *)&(config->height));
    opencl->setKernelArg("moveParticles2", 6, sizeof(float), (void *)&(config->sensorAngle));
    opencl->setKernelArg("moveParticles2", 7, sizeof(float), (void *)&(config->sensorDist));
    opencl->setKernelArg("moveParticles2", 8, sizeof(float), (void *)&(config->rotationAngle));

    opencl->setKernelBufferArg("depositStuff1", 0, "particles");
    opencl->setKernelBufferArg("depositStuff1", 1, "trail");
    opencl->setKernelArg("depositStuff1", 2, sizeof(int), (void *)&(config->width));
    opencl->setKernelArg("depositStuff1", 3, sizeof(int), (void *)&(config->height));
    opencl->setKernelArg("depositStuff1", 4, sizeof(float), (void *)&(config->depositAmount));

    opencl->setKernelBufferArg("depositStuff2", 0, "particles");
    opencl->setKernelBufferArg("depositStuff2", 1, "trailCopy");
    opencl->setKernelArg("depositStuff2", 2, sizeof(int), (void *)&(config->width));
    opencl->setKernelArg("depositStuff2", 3, sizeof(int), (void *)&(config->height));
    opencl->setKernelArg("depositStuff2", 4, sizeof(float), (void *)&(config->depositAmount));

    opencl->setKernelBufferArg("resetTrail", 0, "trail");
    opencl->setKernelBufferArg("resetTrail", 1, "trailCopy");

    opencl->setKernelBufferArg("processTrail1", 0, "trail");
    opencl->setKernelBufferArg("processTrail1", 1, "image");
    opencl->setKernelBufferArg("processTrail1", 2, "colourMap");
    opencl->setKernelArg("processTrail1", 3, sizeof(int), (void *)&(config->num_colours));

    opencl->setKernelBufferArg("processTrail2", 0, "trailCopy");
    opencl->setKernelBufferArg("processTrail2", 1, "image");
    opencl->setKernelBufferArg("processTrail2", 2, "colourMap");
    opencl->setKernelArg("processTrail2", 3, sizeof(int), (void *)&(config->num_colours));

    opencl->setKernelBufferArg("resetImage", 0, "image");

    opencl->setKernelBufferArg("renderParticles", 0, "particles");
    opencl->setKernelBufferArg("renderParticles", 1, "image");
    opencl->setKernelArg("renderParticles", 2, sizeof(int), &(config->width));

    opencl->setKernelBufferArg("invertImage", 0, "image");

    opencl->setKernelBufferArg("lagImage", 0, "image");
    opencl->setKernelBufferArg("lagImage", 1, "image2");

    opencl->setKernelBufferArg("initParticles", 0, "particles");
    opencl->setKernelBufferArg("initParticles", 1, "randomState");
    opencl->setKernelBufferArg("initParticles", 2, "randomIncrement");
    opencl->setKernelArg("initParticles", 3, sizeof(float), &(config->particleStepSize));
    opencl->setKernelArg("initParticles", 4, sizeof(int), &(config->width));
    opencl->setKernelArg("initParticles", 5, sizeof(int), &(config->height));
    
    opencl->setKernelBufferArg("setParticleVels", 0, "particles");
    opencl->setKernelBufferArg("setParticleVels", 1, "randomState");
    opencl->setKernelBufferArg("setParticleVels", 2, "randomIncrement");
    opencl->setKernelArg("setParticleVels", 3, sizeof(float), &(config->particleStepSize));
}

void initPcg() {
    uint64_t *initState, *initSeq;
    initState = (uint64_t *)malloc(config->particleCount * sizeof(uint64_t));
    initSeq = (uint64_t *)malloc(config->particleCount * sizeof(uint64_t));

    for (int i = 0; i < config->particleCount; i++) {
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

    char filename[120] = "colourmaps/default.cm";
    if (strlen(config->colour_file)) {
        fprintf(stderr, "Writing fn\n");
        sprintf(filename, "colourmaps/%s", config->colour_file);
        fprintf(stderr, "Writing fn done\n");
    }
    cm = ColourMapFromFile(filename, config->num_colours);
    cmap = (unsigned int *)malloc(3 * config->num_colours * sizeof(unsigned int));
    cm->apply(cmap);
    opencl->writeBuffer("colourMap", cmap);
    opencl->readBuffer("colourMap", cmap);

    for (int i=0; i < 10; i++) {
        fprintf(stderr, "%d %d %d\n", cmap[3*i], cmap[3*i+1], cmap[3*i+2]);
    }


    pcg32_srandom(time(NULL) ^ (intptr_t)&printf, (intptr_t)&config->particleCount); // Seed pcg
    setKernelArgs();
    initPcg();

    opencl->step("resetTrail");
    opencl->step("initParticles");
}

void cleanup() {
    /* Finalization */
    opencl->cleanup();
}

void iterParticles() {
    if (stepCount % 2 == 0) {
        opencl->step("moveParticles1");
    } else {
        opencl->step("moveParticles2");
    }

    if (stepCount % 2 == 0) {
        opencl->step("depositStuff1");
    } else {
        opencl->step("depositStuff2");
    }
}

void diffuse() {
    if (stepCount % 2 == 0) {
        opencl->step("diffuse1");
    } else {
        opencl->step("diffuse2");
    }
}

void calculateImage() {
    if (settingsMain.renderTrail) {
        if (stepCount % 2 == 0) {
            opencl->step("processTrail1");
        } else {
            opencl->step("processTrail2");
        }
        opencl->readBuffer("image", pixelsMain);
    } else {
        opencl->step("resetImage");
        opencl->step("renderParticles");
        opencl->step("invertImage");
        opencl->step("lagImage");
        opencl->readBuffer("image2", pixelsMain);
    }
}

void step() {
    // iterParticles();
    iterParticles();
    diffuse();
    calculateImage();
    stepCount++;
}

void display() {
    frameCount++;

    if (frameCount % 2 == 0) {
        return;
    }

    opencl->startFrame();

    displayMain();

    if (recording) {
        glReadPixels(0, 0, 1512, 916, GL_RGBA, GL_UNSIGNED_BYTE, buffer);
        fwrite(buffer, sizeof(int) * 1512 * 916, 1, ffmpeg);
    }
    
    // fprintf(stderr, "%.6f                                                    \n", pixelsMain[(int)(config->width * (config->height / 2) + config->width / 2)] / (double)UINT_MAX);
    step();
    iterCount++;

    chrono::high_resolution_clock::time_point temp = chrono::high_resolution_clock::now();
    chrono::duration<float> time_span = chrono::duration_cast<chrono::duration<float>>(temp - timePoint);
    frameTime = time_span.count();
    
    fprintf(stderr, "Step = %d, time = %.4g            \n", frameCount / 2, frameTime);
    fprintf(stderr, "\x1b[%dA", opencl->printCount + 1);
    timePoint = temp;
}

void setupRecording() {
    // start ffmpeg telling it to expect raw rgba 720p-60hz frames
    // -i - tells it to read frames from stdin
    sprintf(cmd, "ffmpeg -r 60 -f rawvideo -pix_fmt rgba -s %dx%d -i - -threads 0 -preset fast -y -pix_fmt yuv420p -crf 21 -vf vflip output.mp4", windowW, windowH);
    // open pipe to ffmpeg's stdin in binary write mode
    ffmpeg = popen(cmd, "w");
    buffer = new int[windowW * windowH];
}

void cleanAll() {
    for (int i = 0; i <= opencl->printCount; i++) {
        fprintf(stderr, "\n");
    }

    fprintf(stderr, "Exiting\n");
    destroyMainWindow();
    opencl->cleanup();

    if (recording) {
        pclose(ffmpeg);
    }
}

int main(int argc, char **argv) {
    if (argc > 1) {
        if (strcmp(argv[1], "-s") == 0) {
            recording = true;
        }
    }

    config = new Config("config.cfg");

    windowW = config->width / 2;
    windowH = config->height / 2;
    size_x_inv = 1. / config->width;
    size_y_inv = 1. / config->height;

    if (recording) {
        setupRecording();
    }

    prepare();
    atexit(&cleanAll);
    
    glutInit(&argc, argv);
    createMainWindow("Physarum", config->width, config->height);
    glutDisplayFunc(&display);
    glutIdleFunc(&display);
    
    glutMainLoop();

    return 0;
}	