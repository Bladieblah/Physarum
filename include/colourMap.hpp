#ifndef COLOURMAP_H
#define COLOURMAP_H

#include <vector>

typedef struct {
    float x;
    float rgb[3];
} ColourFloat;

typedef struct {
    float x;
    unsigned int rgb[3];
} ColourInt;

class ColourMap {
public:
    ColourMap(std::vector<ColourFloat> colours, size_t size, bool symmetric = false);
    ColourMap(std::vector<ColourInt> colours, size_t size, bool symmetric = false);
    
    
    void apply(float *colourMap);
    void apply(unsigned int *colourMap);
    size_t getColorCount();

    std::vector<float> m_x;
    std::vector< std::vector<float> > m_y;
    void generate();
    void save(char *fn);
private:
    size_t m_size;
    size_t m_color_count;
    bool m_symmetric;

    std::vector< std::vector<float> > map;
    std::vector<float> get(float p);
};

ColourMap *ColourMapFromFile(char *fn, size_t size);
extern unsigned int *cmap;
extern ColourMap *cm;

#endif
