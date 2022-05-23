#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>
#include <math.h>
#include <vector>
#include <iostream>

#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <GLUT/glut.h>

#include "colour.hpp"
#include "SimplexNoise.hpp"
#include "pcg.hpp"

// Window size
// #define size_x 1920
// #define size_y 1080
#define size_x 3024
#define size_y 1964
// #define size_x 1400
// #define size_y 801

double size_x_inv = 1. / size_x;
double size_y_inv = 1. / size_y;

bool showColorBar = false;

// Array to be drawn
unsigned int data[size_y*size_x*3];

float *colourMap;
int nColours = 255;

typedef struct Particle {
    double x, y;
    double phi;
} Particle;

int nParticles = 200000;
double sensorAngle = 45. / 180. * M_PI;
double sensorDist = 9;
double rotationAngle = 45. / 180. * M_PI / 18.;
double particleStepSize = 2;
double depositAmount = 0.04;
double stableAverage = 0.1;
double decay = 1 - (nParticles * depositAmount) / (stableAverage * size_x * size_y);
double one_9 = 1. / 9. * decay;

Particle *particles;
double **trail, **trailDummy;

void makeColourmap() {
    // std::vector<float> x = {0., 0.2, 0.4, 0.7, 1.};
    // std::vector< std::vector<float> > y = {
    //     {26,17,36},
    //     {33,130,133},
    //     {26,17,36},
    //     {200,40,187},
    //     {241, 249, 244}
    // };

    std::vector<float> x = {0., 1.};
    std::vector< std::vector<float> > y = {
        {0,0,0},
        {255,255,255}
    };

    Colour col(x, y, nColours);
    
    colourMap = (float *)malloc(3 * nColours * sizeof(float));
    
    col.apply(colourMap);
}

void processTrail() {
    int i, j, k;
    int ind, ind2;
    
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
                ind2 = 3 * (int)(fmin(0.999, trail[i][j]) * nColours);
            
                for (k=0; k<3; k++) {
                    data[ind + k] = colourMap[ind2 + k] * 4294967295;
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
            trail[i][j] = fmin(1., pow((1 + SimplexNoise::noise((i * size_y_inv + seedx) * 10, (j * size_y_inv + seedy) * 10)) / 2, 6) / 2);
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

void initParticles() {
    int i;
    Particle particle;

    for (i = 0; i < nParticles; i++) {
        particle = particles[i];

        particle.x = clip(0.5 + RANDN() * 0.11999, 0.3, 0.7) * size_x;
        particle.y = clip(0.5 + RANDN() * 0.11999, 0.3, 0.7) * size_y;
        particle.phi = 2 * M_PI * UNI();

        particles[i] = particle;
    }
}

void prepare() {
    pcg32_srandom(time(NULL) ^ (intptr_t)&printf, (intptr_t)&nParticles); // Seed pcg

    particles = (Particle *)malloc(nParticles * sizeof(Particle));

    trail = (double **)malloc(size_x * sizeof(double *));
    trailDummy = (double **)malloc(size_x * sizeof(double *));

    for (int i = 0; i < size_x; i++) {
        trail[i] = (double *)malloc(size_y * sizeof(double));
        trailDummy[i] = (double *)malloc(size_y * sizeof(double));
    }

    initData();
    initParticles();
}

void cleanup() {
    /* Finalization */
    free(colourMap);
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

    // if (UNI() < 1.5) {
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
    // }
    // else {
    //     particle->phi += rotationAngle * (UNI() > 0.5 ? 1 : -1);
    // }

    particle->x += cos(particle->phi) * particleStepSize;
    particle->y += sin(particle->phi) * particleStepSize;

    if (particle->x < 0) {
        particle->x += size_x;
    }
    else if (particle->x >= size_x) {
        particle->x -= size_x;
    }

    if (particle->y < 0) {
        particle->y += size_y;
    }
    else if (particle->y >= size_y) {
        particle->y -= size_y;
    }
}

void iterParticles() {
    int i;

    for (i = 0; i < nParticles; i++) {
        // std::cout << "phi: " << particles[i].phi << " y: " << particles[i].y << " // ";
        moveParticle(&(particles[i]));
        // std::cout << "phi: " << particles[i].phi << " y: " << particles[i].y << std::endl;
    }

    for (i = 0; i < nParticles; i++) {
        // Deposit
        trail[(int)particles[i].x][(int)particles[i].y] += depositAmount;
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
    iterParticles();
    diffuse();
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
    
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    step();
    processTrail();
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    fprintf(stderr, "\rStep time = %.4g      ", time_span.count());
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
            initParticles();
            break;
        case 'b':
            showColorBar = !showColorBar;
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
    // std::cout << fmod(-1, size_x) << std::endl;
    fprintf(stderr, "Decay = %.4f\n", decay);
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