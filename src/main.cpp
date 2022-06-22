#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>
#include <math.h>
#include <vector>
#include <iostream>
#include <thread>

#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <GLUT/glut.h>

#include "colour.hpp"
#include "SimplexNoise.hpp"
#include "pcg.hpp"

// Window size
#define size_x 1920
#define size_y 1080

// #define size_x 5120
// #define size_y 2880

// #define size_x 2560
// #define size_y 1440

// #define size_x 6048
// #define size_y 3928

// #define size_x 3024
// #define size_y 1964

// #define size_x 1400
// #define size_y 802

int windowW = 1920;
int windowH = 1080;

double size_x_inv = 1. / size_x;
double size_y_inv = 1. / size_y;

bool showColorBar = false;
bool recording = false;

// Cycling colors
double t = 0;

// Array to be drawn
unsigned int data[size_y*size_x*3];

float *colourMap;
float *colourMap2;
int nColours = 765;

typedef struct Particle {
    double x, y;
    double phi;
    double velocity;
} Particle;

int particleThreads = 6;

int nParticles = 200000;
int particlesPerThread = nParticles / particleThreads;

// Idk
// double sensorAngle = 45. / 180. * M_PI / 80.;
// double sensorDist = 80;
// double rotationAngle = 45. / 180. * M_PI / 18.;
// double particleStepSize = 2;
// double depositAmount = 0.05;
// double stableAverage = 0.3;

// Road network! omg
// double sensorAngle = 0.4732;
// double sensorDist = 26.3819;
// double rotationAngle = 0.1338;
// double particleStepSize = 5.1793;
// double depositAmount = 0.0196;
// double stableAverage = 0.2868;

// Cloudy bu stringy??
// double sensorAngle = 0.5298;
// double sensorDist = 87.6185;
// double rotationAngle = 2.6770;
// double particleStepSize = 4.1530;
// double depositAmount = 0.1068;
// double stableAverage = 0.1107;

// City Grid
// sensorAngle = 0.0474;
// sensorDist = 14.9574;
// rotationAngle = 0.2148;
// particleStepSize = 2.4507;
// depositAmount = 0.0284;
// stableAverage = 0.1357;

// Double highway
// double sensorAngle = 1.0376;
// double sensorDist = 40.5436;
// double rotationAngle = 0.1885;
// double particleStepSize = 8.7137;
// double depositAmount = 0.0571;
// double stableAverage = 0.1194;

// More roads
double sensorAngle = 0.6659;
double sensorDist = 52.7099;
double rotationAngle = 0.3576;
double particleStepSize = 3.2715;
double depositAmount = 0.0404;
double stableAverage = 0.2899;

// Double highway with spikes
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
// sensorAngle = 0.3596;
// sensorDist = 158.0992;
// rotationAngle = 0.2333;
// particleStepSize = 6.0674;
// depositAmount = 0.0099;
// stableAverage = 0.3276;

// Gridsss
// sensorAngle = 0.5191;
// sensorDist = 29.3579;
// rotationAngle = 0.4350;
// particleStepSize = 8.8227;
// depositAmount = 0.2609;
// stableAverage = 0.3571;

// Ropey
// double sensorAngle = 45. / 180. * M_PI / 8.;
// double sensorDist = 20;
// double rotationAngle = 45. / 180. * M_PI / 18.;
// double particleStepSize = 2;
// double depositAmount = 0.01;
// double stableAverage = 0.2;

// Supernova
// double sensorAngle = -45. / 180. * M_PI / 80.;
// double sensorDist = 200;
// double rotationAngle = 45. / 180. * M_PI / 180.;
// double particleStepSize = 2;
// double depositAmount = 0.001;
// double stableAverage = 0.2;

// Stringy
// double sensorAngle = 45. / 180. * M_PI / 8.;
// double sensorDist = 20;
// double rotationAngle = 45. / 180. * M_PI / 18.;
// double particleStepSize = 9;
// double depositAmount = 0.01;
// double stableAverage = 0.2;

// Firey
// double sensorAngle = 45. / 180. * M_PI / 2.;
// double sensorDist = 10;
// double rotationAngle = -45. / 180. * M_PI / 6.;
// double particleStepSize = 2;
// double depositAmount = 0.1;
// double stableAverage = 0.3;

double decay = 1 - (nParticles * depositAmount) / (stableAverage * size_x * size_y);
double one_9 = 1. / 9. * decay;

Particle **particles;
double **trail, **trailDummy;

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

    std::vector<float> x_w = {0., 1.};
    std::vector< std::vector<float> > y_w = {
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
    // colourMap2 = (float *)malloc(3 * nColours * sizeof(float));
    
    col.apply(colourMap);
    // col2.apply(colourMap2);
}

double sigmoid(double x) {
    return 1. / (1. + exp(-x));
}

double rescaleFactor = sqrt(0.7) / sigmoid(0.7);
double invsqrt07 = 1. / sqrt(0.7);

double rescaleTrail(double x) {
    if (x < 0.7) {
        return sqrt(x);
    }

    return sigmoid(x) * rescaleFactor;
}

double rescaleTrail2(double x) {
    if (x < 0.7) {
        return sqrt(x) * invsqrt07;
    }

    return 0.5 + cos(sqrt(x - 0.7)) * 0.5;
}

void processTrail() {
    int i, j, k;
    int ind, ind2;

    double ct2 = pow(cos(t), 2);
    double st2 = 1 - ct2;
    
    for (i=0; i<size_x; i++) {
        for (j=0; j<size_y; j++) {
            ind = 3 * (size_x * j + i);
            if (showColorBar && i < 70) {
                int colInd = 3 * (int)(j / (double)size_y * nColours);
                data[ind + 0] = colourMap[colInd + 0] * 4294967295;
                data[ind + 1] = colourMap[colInd + 1] * 4294967295;
                data[ind + 2] = colourMap[colInd + 2] * 4294967295;
            }
            else {
                ind2 = 3 * (int)(fmin(0.999, rescaleTrail2(trail[i][j])) * nColours);
            
                for (k=0; k<3; k++) {
                    data[ind + k] = colourMap[ind2 + k] * 4294967295;
                    // data[ind + k] = (colourMap[ind2 + k] * ct2 + colourMap2[ind2 + k] * st2) * 4294967295;
                }
            }
        }
    }
}

void initData() {
    int i,j;

    int xc = size_x / 2;
    int yc = size_y / 2;

    int R = size_y / 4;
    double r, w = 5;

    double seedx = UNI() * 20 - 10;
    double seedy = UNI() * 20 - 10;

    for (i = 0; i < size_x; i++) {
        for (j = 0; j < size_y; j++) {
            r = sqrt(pow(i - xc, 2) + pow(j - yc, 2));
            // trail[i][j] = fmin(1., exp(-pow((r - R) / w, 2)) + pow((1 + SimplexNoise::noise(i * size_y_inv * 60, j * size_y_inv * 60)) / 2, 6) / 2);
            // trail[i][j] = fmin(1., pow((1 + SimplexNoise::noise((i * size_y_inv + seedx) * 10, (j * size_y_inv + seedy) * 10)) / 2, 6) / 2);
            trail[i][j] = 0;
        }
    }
}

double clip(double in, double lower, double upper) {
    if (in < lower) {
        return lower;
    }
    if (in > upper) {
        return upper;
    }

    return in;
}

void initParticles(int thread) {
    int i;
    Particle particle;

    // Squaretangle
    // for (i = 0; i < particlesPerThread; i++) {
    //     particle = particles[thread][i];

    //     particle.x = clip(0.5 + RANDN() * 0.31999, 0.3, 0.7) * size_x;
    //     particle.y = clip(0.5 + RANDN() * 0.31999, 0.3, 0.7) * size_y;
    //     particle.phi = 2 * M_PI * UNI();

    //     particles[thread][i] = particle;
    // }

    // Circle
    double xc = size_x * 0.5;
    double yc = size_y * 0.5;
    for (i = 0; i < particlesPerThread; i++) {
        particle = particles[thread][i];

        double theta = UNI() * 2 * M_PI;
        double rad = (RANDN() / 8. + 0.05);

        particle.x = clip(cos(theta) * rad * size_y + xc, 0., size_x);
        particle.y = clip(sin(theta) * rad * size_y + yc, 0., size_y);
        particle.phi = pow(atan2(yc - particle.y, xc - particle.x), 2.) + UNI();
        particle.velocity = UNI() * particleStepSize + 1;

        particles[thread][i] = particle;
    }
}

void prepare() {
    pcg32_srandom(time(NULL) ^ (intptr_t)&printf, (intptr_t)&nParticles); // Seed pcg

    particles = (Particle **)malloc(particleThreads * sizeof(Particle *));

    for (int i = 0; i < particleThreads; i++) {
        particles[i] = (Particle *)malloc(particlesPerThread * sizeof(Particle));
        initParticles(i);
    }

    trail = (double **)malloc(size_x * sizeof(double *));
    trailDummy = (double **)malloc(size_x * sizeof(double *));

    for (int i = 0; i < size_x; i++) {
        trail[i] = (double *)malloc(size_y * sizeof(double));
        trailDummy[i] = (double *)malloc(size_y * sizeof(double));
    }

    initData();
}

void cleanup() {
    /* Finalization */
    free(colourMap);

    for (int i = 0; i < particleThreads; i++) {
        free(particles[i]);
    }
    free(particles);

    for (int i = 0; i < size_x; i++) {
        free(trail[i]);
        free(trailDummy[i]);
    }
    free(trail);
    free(trailDummy);
}

void moveParticle(Particle *particle) {
    double fl, fc, fr;
    double flx, fly, fcx, fcy, frx, fry;

    if (UNI() < 0.99) {
        flx = particle->x + cos(particle->phi - sensorAngle) * sensorDist;
        fly = particle->y + sin(particle->phi - sensorAngle) * sensorDist;

        fcx = particle->x + cos(particle->phi) * sensorDist;
        fcy = particle->y + sin(particle->phi) * sensorDist;

        frx = particle->x + cos(particle->phi + sensorAngle) * sensorDist;
        fry = particle->y + sin(particle->phi + sensorAngle) * sensorDist;

        fl = trail[(int)(flx + size_x) % size_x][(int)(fly + size_y) % size_y];
        fc = trail[(int)(fcx + size_x) % size_x][(int)(fcy + size_y) % size_y];
        fr = trail[(int)(frx + size_x) % size_x][(int)(fry + size_y) % size_y];

        if (fc < fl && fc < fr) {
            particle->phi += rotationAngle * (UNI() > 0.5 ? 1 : -1);
        }
        else if (fl > fc && fc > fr) {
            particle->phi -= rotationAngle;
        }
        else if (fl < fc && fc < fr) {
            particle->phi += rotationAngle;
        }
    }
    else {
        particle->phi += 10 * rotationAngle * (UNI() > 0.5 ? 1 : -1);
    }

    particle->x += cos(particle->phi) * particle->velocity;
    particle->y += sin(particle->phi) * particle->velocity;

    if (particle->x < 0) {
        particle->x = -particle->x;
        particle->phi = M_PI - particle->phi;
    } else if (particle->x >= size_x) {
        particle->x = 2 * size_x - particle->x;
        particle->phi = M_PI - particle->phi;
    }

    if (particle->y < 0) {
        particle->y = -particle->y;
        particle->phi = -particle->phi;
    } else if (particle->y >= size_y){
        particle->y = 2 * size_y - particle->y;
        particle->phi = -particle->phi;
    }

    // if (particle->x < 0) {
    //     particle->x += size_x;
    // }
    // else if (particle->x >= size_x) {
    //     particle->x -= size_x;
    // }

    // if (particle->y < 0) {
    //     particle->y += size_y;
    // }
    // else if (particle->y >= size_y) {
    //     particle->y -= size_y;
    // }
}

void moveParticles(int thread) {
    int i;

    for (i = 0; i < particlesPerThread; i++) {
        moveParticle(&(particles[thread][i]));
    }
}

void depositStuff(int thread) {
    int i;

    for (i = 0; i < particlesPerThread; i++) {
        trail[(int)particles[thread][i].x][(int)particles[thread][i].y] += depositAmount;
    }
}

void iterParticles() {
    int i;
	std::thread *tt = new std::thread[particleThreads - 1];

    for (i = 0; i < particleThreads - 1; i++) {
		tt[i] = std::thread(moveParticles, i);
	}
	moveParticles(particleThreads - 1);
	for (i = 0; i < particleThreads - 1; ++i) {
		tt[i].join();
	}

    for (i = 0; i < particleThreads - 1; i++) {
		tt[i] = std::thread(depositStuff, i);
	}
	depositStuff(particleThreads - 1);
	for (i = 0; i < particleThreads - 1; ++i) {
		tt[i].join();
	}
}

void diffuse() {
    int i, j, k, l, km, lm;
    double conv;

    for (i = 0; i < size_x; i++) {
        for (j = 0; j < size_y; j++) {
            conv = 0;

            for (k = -1; k < 2; k++) {
                km = (i + k + size_x) % size_x;
                for (l = -1; l < 2; l++) {
                    lm = (j + l + size_y) % size_y;
                    conv += trail[km][lm];
                }
            }

            trailDummy[i][j] = conv * one_9;
        }
    }

    for (i = 0; i < size_x; i++) {
        for (j = 0; j < size_y; j++) {
            trail[i][j] = trailDummy[i][j];
        }
    }
}

void step() {
    // iterParticles();
    iterParticles();
    diffuse();
    t += 0.0015;
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

    if (recording && t > 0) {
        glReadPixels(0, 0, windowW, windowH, GL_RGBA, GL_UNSIGNED_BYTE, buffer);
        fwrite(buffer, sizeof(int)*windowW*windowH, 1, ffmpeg);
    }
    
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    step();
    processTrail();
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    fprintf(stderr, "\rStep time = %.4g      ", time_span.count());
}

void randomiseParameters() {
    sensorAngle = 2 * UNI() * M_PI;
    sensorDist = UNI() * 200;
    rotationAngle = 2 * UNI() * M_PI;
    particleStepSize = UNI() * 10;
    depositAmount = exp(UNI() * 4 - 5.5);
    stableAverage = UNI() * 0.3 + 0.1;

    decay = 1 - (nParticles * depositAmount) / (stableAverage * size_x * size_y);
    one_9 = 1. / 9. * decay;

    fprintf(stderr, "\nsensorAngle = %.4f;\nsensorDist = %.4f;\nrotationAngle = %.4f;\nparticleStepSize = %.4f;\ndepositAmount = %.4f;\nstableAverage = %.4f;\n\n",
        sensorAngle, sensorDist, rotationAngle, particleStepSize, depositAmount, stableAverage);
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
        case 'i':
            for (int i = 0; i < particleThreads; i++){
                initParticles(i);
            }
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
    sprintf(cmd, "ffmpeg -r 60 -f rawvideo -pix_fmt rgba -s %dx%d -i - -threads 0 -preset fast -y -pix_fmt yuv420p -crf 21 -vf vflip output.mp4", windowW, windowH);
    // open pipe to ffmpeg's stdin in binary write mode
    ffmpeg = popen(cmd, "w");
    buffer = new int[windowW*windowH];
}

int main(int argc, char **argv) {
    prepare();

    makeColourmap();
	
    glutInit( &argc, argv );
    glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE );
    glutInitWindowSize( windowW, windowH );
    glutCreateWindow( "Physarum" );
    glutDisplayFunc( display );
    
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