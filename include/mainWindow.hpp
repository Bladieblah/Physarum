
#ifndef RENDER_WINDOW_H
#define RENDER_WINDOW_H

#include "config.hpp"
#include "opencl.hpp"

typedef struct WindowSettings {
    uint32_t width, height;
    uint32_t windowW, windowH;
    float zoom = 1, centerX = 0, centerY = 0;
    bool grid = false;
    bool renderTrail = false;
} WindowSettings;

typedef struct MouseState {
    int xDown, yDown;
    int x, y;
    int state = 1; // GLUT_UP
} MouseState;

void displayMain();

void createMainWindow(char *name, uint32_t width, uint32_t height);
void destroyMainWindow();

extern WindowSettings settingsMain;
extern uint32_t *pixelsMain;

extern float frameTime;
extern uint32_t iterCount;

#endif
