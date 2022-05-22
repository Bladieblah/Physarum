#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>
#include <math.h>
#include <vector>

#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <GLUT/glut.h>

#include "colour.hpp"

// Window size
// #define size_x 1920
// #define size_y 1080
#define size_x 1400
#define size_y 801

bool showColorBar = true;
double one_9 = 1. / 9.;

// Array to be drawn
unsigned int data[size_y*size_x*3];

double diffKernel[3][3] = {
    {1./9., 1./9., 1./9.},
    {1./9., 1./9., 1./9.},
    {1./9., 1./9., 1./9.}
};

float *colourMap;
int nColours = 255;

typedef struct Particle {
    double x, y;
    double phi;
} Particle;

int nParticles = 10000;

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
    double r, w = 15;

    for (i = 0; i < size_x; i++) {
        for (j = 0; j < size_y; j++) {
            r = sqrt(pow(i - xc, 2) + pow(j - yc, 2));
            trail[i][j] = exp(-pow((r - R) / w, 2));
        }
    }
}

void prepare() {
    particles = (Particle *)malloc(nParticles * sizeof(Particle));

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
    free(particles);

    for (int i = 0; i < size_x; i++) {
        free(trail[i]);
        free(trailDummy[i]);
    }
    free(trail);
    free(trailDummy);
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
    diffuse();
    processTrail();
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