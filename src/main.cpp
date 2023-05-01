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
#include "config.hpp"
#include "SimplexNoise.hpp"
#include "pcg.hpp"
#include "opencl.hpp"

using namespace std;


int windowW, windowH;

float size_x_inv, size_y_inv;

bool showColorBar = false;
bool recording = false;
bool renderTrail = false;

// Cycling colors
double t = 0;

// Array to be drawn
uint32_t *pixelData;

float *colourMap;
float *colourMap2;
int nColours = 765;

typedef struct Particle {
    float x, y;
    float phi;
    float velocity;
} Particle;


// OpenCl stuff

OpenCl *opencl;
Config *config;
uint frameCount = 0;
uint stepCount = 0;

vector<BufferSpec> bufferSpecs;
void createBufferSpecs() {
    bufferSpecs = {
        {"trail",     {NULL, config->width * config->height * sizeof(float)}},
        {"trailCopy", {NULL, config->width * config->height * sizeof(float)}},
        {"particles", {NULL, config->particleCount * sizeof(Particle)}},
        {"random",    {NULL, (config->particleCount + 2) * sizeof(float)}},
        {"image",     {NULL, 3 * config->width * config->height * sizeof(uint32_t)}},
        {"image2",    {NULL, 3 * config->width * config->height * sizeof(uint32_t)}},
        {"colourMap", {NULL, 3 * nColours * sizeof(float)}},

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

    std::vector<float> x_bg = {0., 0.25, 0.5, 0.75, 1.};
    std::vector< std::vector<float> > y_bg = {
        {36,35,49},
        // {29,106,154},
        // {15,171,179},
        // {36,153,120},
        // {65,175,131},
        {21,190,5},
    };

    std::vector<float> x_gb = {0., 0.25, 0.5, 0.75, 1.};
    std::vector< std::vector<float> > y_gb = {
        {36,35,49},
        // {44,115,33},
        // {74,162,37},
        // {62,121,103},
        // {43,171,137},
        {16,185,193},
    };

    Colour col(x_i, y_i, nColours);

    // Colour col(x_w, y_bg, nColours);
    // Colour col2(x_w, y_gb, nColours);
    
    colourMap = (float *)malloc(3 * nColours * sizeof(float));
    col.apply(colourMap);

    opencl->writeBuffer("colourMap", (void *)colourMap);
}

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
    opencl->setKernelArg("processTrail1", 3, sizeof(int), (void *)&nColours);

    opencl->setKernelBufferArg("processTrail2", 0, "trailCopy");
    opencl->setKernelBufferArg("processTrail2", 1, "image");
    opencl->setKernelBufferArg("processTrail2", 2, "colourMap");
    opencl->setKernelArg("processTrail2", 3, sizeof(int), (void *)&nColours);

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

    pcg32_srandom(time(NULL) ^ (intptr_t)&printf, (intptr_t)&config->particleCount); // Seed pcg
    pixelData = (uint32_t *)malloc(config->height * config->width * 3 * sizeof(uint32_t));

    setKernelArgs();

    initPcg();
    makeColourmap();

    opencl->step("resetTrail");
    opencl->step("initParticles");
}

void cleanup() {
    /* Finalization */
    free(colourMap);
    free(pixelData);

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
    // iterParticles();
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
            config->width,
            config->height,
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
        fprintf(stderr, "Step = %d, time = %.4g               \n", frameCount / 2, time_span.count());
        if (renderTrail) {
            fprintf(stderr, "\x1b[5A");
        } else {
            fprintf(stderr, "\x1b[8A");
        }

    }

    frameCount++;
}

void randomiseParameters() {
    config->sensorAngle = 2 * UNI() * M_PI;
    config->sensorDist = UNI() * 200;
    config->rotationAngle = 2 * UNI() * M_PI;
    config->particleStepSize = UNI() * 10;
    config->depositAmount = exp(UNI() * 4 - 5.5);
    config->stableAverage = UNI() * 0.3 + 0.1;

    float decayFactor = 1 - (config->particleCount * config->depositAmount) / (config->stableAverage * config->width * config->height);
    float one_9 = 1. / 9. * decayFactor;

    fprintf(stderr, "\n\n\n\n\n\n\n\n\nsensorAngle = %.4f;\nsensorDist = %.4f;\nrotationAngle = %.4f;\nparticleStepSize = %.4f;\ndepositAmount = %.4f;\nstableAverage = %.4f;\n\n",
        config->sensorAngle, config->sensorDist, config->rotationAngle, config->particleStepSize, config->depositAmount, config->stableAverage);

    opencl->setKernelArg("diffuse1", 2, sizeof(float), (void *)&one_9);
    opencl->setKernelArg("moveParticles1", 6, sizeof(float), (void *)&(config->sensorAngle));
    opencl->setKernelArg("moveParticles1", 7, sizeof(float), (void *)&(config->sensorDist));
    opencl->setKernelArg("moveParticles1", 8, sizeof(float), (void *)&(config->rotationAngle));
    opencl->setKernelArg("depositStuff1",  4, sizeof(float), (void *)&(config->depositAmount));

    opencl->setKernelArg("diffuse2", 2, sizeof(float), (void *)&one_9);
    opencl->setKernelArg("moveParticles2", 6, sizeof(float), (void *)&(config->sensorAngle));
    opencl->setKernelArg("moveParticles2", 7, sizeof(float), (void *)&(config->sensorDist));
    opencl->setKernelArg("moveParticles2", 8, sizeof(float), (void *)&(config->rotationAngle));
    opencl->setKernelArg("depositStuff2",  4, sizeof(float), (void *)&(config->depositAmount));

    opencl->setKernelArg("setParticleVels", 3, sizeof(float), &(config->particleStepSize));
    opencl->step("setParticleVels");
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
            opencl->step("resetTrail");
            break;
        case 'i':
            opencl->step("initParticles");
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
    sprintf(cmd, "ffmpeg -r 60 -f rawvideo -pix_fmt rgba -s %dx%d -i - -threads 0 -preset fast -y -pix_fmt yuv420p -crf 21 -vf vflip output.mp4", windowW, windowH);
    // open pipe to ffmpeg's stdin in binary write mode
    ffmpeg = popen(cmd, "w");
    buffer = new int[windowW*windowH];
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

    prepare();

    makeColourmap();
	
    glutInit( &argc, argv );
    glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE );
    glutInitWindowSize( config->width, config->height );
    glutCreateWindow( "Physarum" );
    
    glutDisplayFunc(&display);
    glutIdleFunc(&display);
    glutKeyboardUpFunc(&key_pressed);
    glutReshapeFunc(&reshape);

    if (argc > 1) {
        if (strcmp(argv[1], "-s") == 0) {
            recording = true;
        }
    }

    if (recording)
        setupRecording();
    
    glutMainLoop();

    return 0;
}	