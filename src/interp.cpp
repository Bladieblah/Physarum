#include <vector>

#include "interp.hpp"

using namespace std;

Interp1d::Interp1d(vector<float> _x, vector< vector<float> > _y) {
    for (size_t i = 0; i < _x.size(); i++) {
        vector<float> temp(_y[i]);
        
        m_x.push_back(_x[i]);
        m_y.push_back(temp);
    }
}

float Interp1d::interpolate(float x0, float x1, float y0, float y1, float p) {
    return y0 + (y1 - y0) / (x1 - x0) * (p - x0);
}

vector<float> Interp1d::getValue(float p) {
    size_t i, j;
    vector<float> result;
    
    if (p >= m_x[(int)m_x.size() - 1]) {
        return (vector<float>)(m_y[m_x.size() - 1]);
    }
    
    for (i = 0; i < m_x.size(); i++) {
        if (p < m_x[i])
            break;
    }
    
    if (i == 0) {
        return (vector<float>)(m_y[0]);
    }
    
    for (j = 0; j < m_y[i].size(); j++) {
        result.push_back(interpolate(m_x[i-1], m_x[i], m_y[i-1][j], m_y[i][j], p));
    }
    
    return result;
}