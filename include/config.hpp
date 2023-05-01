#ifndef CONFIG_H
#define CONFIG_H

#include <map>
#include <string>

typedef struct Setting {
    char type;
    void *pointer;
} Setting;

class Config {
public:
    unsigned int particleCount = 1000000;

    unsigned int width = 1080;
    unsigned int height = 720;

    float sensorAngle = 0.4367;
    float sensorDist = 13.2189;
    float rotationAngle = 0.7218;
    float particleStepSize = 5.9052;
    float depositAmount = 0.1264;
    float stableAverage = 0.2992;

    bool profile = true;
    bool verbose = true;

    Config(char *filename);
    void printValues();

private:
    void processLine(std::string line);
    void setValue(std::string name, char *value);
    void printSetting(std::string name, Setting setting);

    const std::map<std::string, Setting> typeMap = {
        {"particleCount", {'i', (void *)&particleCount}},

        {"width", {'i', (void *)&width}},
        {"height", {'i', (void *)&height}},

        {"sensorAngle",      {'f', (void *)&sensorAngle}},
        {"sensorDist",       {'f', (void *)&sensorDist}},
        {"rotationAngle",    {'f', (void *)&rotationAngle}},
        {"particleStepSize", {'f', (void *)&particleStepSize}},
        {"depositAmount",    {'f', (void *)&depositAmount}},
        {"stableAverage",    {'f', (void *)&stableAverage}},
        
        {"profile", {'b', (void *)&profile}},
        {"verbose", {'b', (void *)&verbose}},
    };
};

extern Config *config;

#endif
