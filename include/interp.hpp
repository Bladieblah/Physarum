#ifndef INTERP_H
#define INTERP_H

#include <math.h>
#include <vector>

class Interp1d {
public:
    Interp1d(std::vector<float> _x, std::vector< std::vector<float> > _y);
    
    float interpolate(float x0, float x1, float y0, float y1, float p);
    std::vector<float> getValue(float p);
    
    std::vector<float> m_x;
    std::vector< std::vector<float> > m_y;
};

#endif
