#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <stack>

#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <GLUT/glut.h>

#include "../imgui/imgui.h"
#include "../imgui/backends/imgui_impl_glut.h"
#include "../imgui/backends/imgui_impl_opengl2.h"

#include "config.hpp"
#include "mainWindow.hpp"
#include "opencl.hpp"
#include "pcg.hpp"

int windowIdMain;
uint32_t *pixelsMain;

WindowSettings settingsMain;
MouseState mouseMain;

bool showColorBar = false;

void drawGrid() {
    glBegin(GL_LINES);
        glVertex2f(-1,0); glVertex2f(1,0);
        glVertex2f(-1,0.5); glVertex2f(1,0.5);
        glVertex2f(-1,-0.5); glVertex2f(1,-0.5);
        glVertex2f(0,-1); glVertex2f(0,1);
        glVertex2f(0.5,-1); glVertex2f(0.5,1);
        glVertex2f(-0.5,-1); glVertex2f(-0.5,1);
    glEnd();
}

void showInfo() {
    ImGui::SeparatorText("Info");
    ImGui::Text("Frametime = %.3f", frameTime);
    ImGui::Text("Iterations = %d", iterCount);
}

void displayMain() {
    // --------------------------- RESET ---------------------------
    glutSetWindow(windowIdMain);

    ImGuiIO& io = ImGui::GetIO();

    glClearColor(0, 0, 0, 1);
    glColor3f(1, 1, 1);
    glClear( GL_COLOR_BUFFER_BIT );

    glEnable (GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    
    // --------------------------- FRACTAL ---------------------------

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
        &pixelsMain[0]
    );

    glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0, -1.0);
        glTexCoord2f(1.0f, 0.0f); glVertex2f( 1.0, -1.0);
        glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.0,  1.0);
        glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0,  1.0);
    glEnd();

    if (settingsMain.grid) {
        drawGrid();
    }

    // --------------------------- IMGUI ---------------------------

    ImGui_ImplOpenGL2_NewFrame();
    ImGui_ImplGLUT_NewFrame();
    ImGui::NewFrame();

    ImGui::SetNextWindowSize(ImVec2(220, 0));
    ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x - 200, 0));

    ImGui::Begin("Gradient colors");
    ImGui::PushItemWidth(140);

    showInfo();

    ImGui::End();

    // --------------------------- DRAW ---------------------------
    ImGui::Render();
    ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());
    glFlush();
    glutSwapBuffers();
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

void keyPressedMain(unsigned char key, int x, int y) {
    switch (key)
    {
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
            settingsMain.renderTrail = !settingsMain.renderTrail;
            break;
        case 'u':
            randomiseParameters();
            break;
        case 'q':
            exit(0);
            break;
        default:
            break;
    }
}

void specialKeyPressedMain(int key, int x, int y) {
    ImGui_ImplGLUT_SpecialFunc(key, x, y);
}

void mousePressedMain(int button, int state, int x, int y) {
    ImGuiIO& io = ImGui::GetIO();
    ImGui_ImplGLUT_MouseFunc(button, state, x, y);

    if (!io.WantCaptureMouse && button == GLUT_RIGHT_BUTTON) {
        return;
    }

    mouseMain.state = state;

    if (state == GLUT_DOWN) {
        mouseMain.xDown = x;
        mouseMain.yDown = y;
    }
}

void mouseMovedMain(int x, int y) {
    ImGuiIO& io = ImGui::GetIO();
    ImGui_ImplGLUT_MotionFunc(x, y);

    mouseMain.x = x;
    mouseMain.y = y;
}

void onReshapeMain(int w, int h) {
    ImGui_ImplGLUT_ReshapeFunc(w, h);

    settingsMain.windowW = w;
    settingsMain.windowH = h;

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glViewport(0, 0, w, h);
    glMatrixMode(GL_MODELVIEW);
}

void createMainWindow(char *name, uint32_t width, uint32_t height) {
    settingsMain.width = width;
    settingsMain.height = height;

    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    windowIdMain = glutCreateWindow(name);

    pixelsMain = (uint32_t *)malloc(3 * width * height * sizeof(uint32_t));

    for (int i = 0; i < 3 * width * height; i++) {
        pixelsMain[i] = 0;
    }

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO(); (void)io;
    io.IniFilename = NULL;

    ImGui::StyleColorsDark();
    ImGui_ImplGLUT_Init();
    ImGui_ImplOpenGL2_Init();

    glutKeyboardUpFunc(ImGui_ImplGLUT_KeyboardUpFunc);
    glutSpecialUpFunc(ImGui_ImplGLUT_SpecialUpFunc);

    glutKeyboardFunc(&keyPressedMain);
    glutSpecialFunc(&specialKeyPressedMain);
    glutMouseFunc(&mousePressedMain);
    glutMotionFunc(&mouseMovedMain);
    glutPassiveMotionFunc(&mouseMovedMain);
    glutReshapeFunc(&onReshapeMain);

    glutDisplayFunc(&displayMain);
}

void destroyMainWindow() {
    free(pixelsMain);
}
