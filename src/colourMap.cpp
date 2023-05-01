#include <vector>

#include "colourMap.hpp"
#include "interp.hpp"

using namespace std;


vector<ColourInt> defaultColours = {
    {0.0, {0, 10, 2}},
    {0.05, {3, 34, 12}},
    {0.2, {8, 76, 33}},
    {0.4, {13, 117, 53}},
    {0.7, {10, 172, 102}},
    {0.9, {63, 216, 112}},
    {1.0, {25, 236, 173}},
};

// --------------------------- IO ---------------------------

ColourMap *ColourMapFromInt(FILE *f, size_t size, bool symmetric) {
    ColourInt tmp;
    vector<ColourInt> colours;

    while (fscanf(f, "%f, {%d, %d, %d}\n", &(tmp.x), &(tmp.rgb[0]), &(tmp.rgb[1]), &(tmp.rgb[2])) == 4) {
        colours.push_back(ColourInt(tmp));
    }

    fclose(f);

    return new ColourMap(colours, size, symmetric);
}

ColourMap *ColourMapFromFloat(FILE *f, size_t size, bool symmetric) {
    ColourFloat tmp;
    vector<ColourFloat> colours;

    while (fscanf(f, "%f, {%f, %f, %f}\n", &(tmp.x), &(tmp.rgb[0]), &(tmp.rgb[1]), &(tmp.rgb[2])) == 4) {
        colours.push_back(ColourFloat(tmp));
    }

    fclose(f);

    return new ColourMap(colours, size, symmetric);
}

ColourMap *ColourMapFromFile(char *fn, size_t size) {
    FILE *f;
    ColourMap *colourmap;
    char kind = 'a';
    int symmetric = 0;

    fprintf(stderr, "Loading %s\n", fn);

    f = fopen(fn, "r");

    if (!f) {
        fprintf(stderr, "Error loading cm.\n");
        return new ColourMap(defaultColours, size, false);
    }

    if (fscanf(f, "kind = %c\n", &kind) == EOF || fscanf(f, "symmetric = %d\n", &symmetric) == EOF) {
        fprintf(stderr, "Error loading cm.\n");
        colourmap = new ColourMap(defaultColours, size, false);
    } else {
        switch (kind) {
            case 'i':
                colourmap = ColourMapFromInt(f, size, symmetric);
                break;
            case 'f':
                colourmap = ColourMapFromFloat(f, size, symmetric);
                break;
            default:
                colourmap = new ColourMap(defaultColours, size, true);
                break;
        }
    }

    fclose(f);

    return colourmap;
}

// --------------------------- Class implementation ---------------------------

ColourMap::ColourMap(vector<ColourFloat> colours, size_t size, bool symmetric) {
    m_size = size;
    m_color_count = colours.size();
    m_symmetric = symmetric;

    map.reserve(m_size);

    for (ColourFloat colour : colours) {
        vector<float> tmp;
        tmp.push_back(colour.rgb[0]);
        tmp.push_back(colour.rgb[1]);
        tmp.push_back(colour.rgb[2]);
        m_x.push_back(colour.x);
        m_y.push_back(tmp);
    }
    
    generate();
}

ColourMap::ColourMap(vector<ColourInt> colours, size_t size, bool symmetric) {
    m_size = size;
    m_color_count = colours.size();
    m_symmetric = symmetric;

    map.resize(m_size);

    for (ColourInt colour : colours) {
        vector<float> tmp;
        for (size_t i = 0; i < 3; i++) {
            tmp.push_back((float)colour.rgb[i] / 255.);
        }
        
        m_x.push_back(colour.x);
        m_y.push_back(tmp);
    }
    
    generate();
}

void ColourMap::generate() {
    Interp1d interp(m_x, m_y);
    
    float p = 0;
    float dp = 1. / (m_size - 1.);
    
    for (size_t i = 0; i < m_size; i++) {
        vector<float> result = interp.getValue(m_symmetric ? 1 - fabs(2 * p - 1) : p);
        map[i] = result;
        
        p += dp;
    }
}

vector<float> ColourMap::get(float p) {
    size_t i = (size_t)(p * m_size);
    
    return map[i];
}

void ColourMap::apply(float *colourMap) {
    for (size_t i = 0; i < m_size; i++) {
        for (size_t j = 0; j < 3; j++) {
            colourMap[3 * i + j] = map[i][j];
        }
    }
}

void ColourMap::apply(unsigned int *colourMap) {
    for (size_t i = 0; i < m_size; i++) {
        for (size_t j = 0; j < 3; j++) {
            colourMap[3 * i + j] = (unsigned int)(map[i][j] * UINT_MAX);
        }
    }
}

size_t ColourMap::getColorCount() {
    return m_color_count;
}

void ColourMap::save(char *fn) {
    fprintf(stderr, "Saving to %s          \n", fn);
    FILE *outFile = fopen(fn, "w");
    
    if (outFile) {
        fprintf(outFile, "kind = i\n");
        fprintf(outFile, "symmetric = %d\n", (int)m_symmetric);

        for (int i = 0; i < m_color_count; i++) {
            fprintf(outFile, "%f, {%d, %d, %d}\n", m_x[i], (int)(m_y[i][0] * 255), (int)(m_y[i][1] * 255), (int)(m_y[i][2] * 255));
        }

        fclose(outFile);
    }
}
