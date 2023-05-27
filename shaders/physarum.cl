#define PERIODIC_BOUNDARY

__constant sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_NEAREST | CLK_ADDRESS_NONE;

__constant float rescaleFactor = 1.25213309998062819936804151;
__constant float invsqrt07 = 1.19522860933439363996881717;
__constant float imageLim = 0.999;

/**
 * RNG stuff
 */

__constant ulong PCG_SHIFT = 6364136223846793005ULL;
__constant float PCG_MAX_1 = 4294967296.0;

#define UNIFORM_LOW 0.02425
#define UNIFORM_HIGH 0.97575

__constant float a[] = {
    -3.969683028665376e+01,
     2.209460984245205e+02,
    -2.759285104469687e+02,
     1.383577518672690e+02,
    -3.066479806614716e+01,
     2.506628277459239e+00
};

__constant float b[] = {
    -5.447609879822406e+01,
     1.615858368580409e+02,
    -1.556989798598866e+02,
     6.680131188771972e+01,
    -1.328068155288572e+01
};

__constant float c[] = {
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e+00,
    -2.549732539343734e+00,
     4.374664141464968e+00,
     2.938163982698783e+00
};

__constant float d[] = {
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e+00,
    3.754408661907416e+00
};

inline float inverseNormalCdf(float u) {
    float q, r;

    if (u <= 0) {
        return -HUGE_VAL;
    }
    else if (u < UNIFORM_LOW) {
        q = sqrt(-2 * log(u));
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    }
    else if (u <= UNIFORM_HIGH) {
        q = u - 0.5;
        r = q * q;
        
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
    }
    else if (u < 1) {
        q  = sqrt(-2 * log(1 - u));

        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    }
    else {
        return HUGE_VAL;
    }
}

inline ulong pcg32Random(global ulong *randomState, global ulong *randomIncrement, int x) {
    ulong oldstate = randomState[x];
    randomState[x] = oldstate * PCG_SHIFT + randomIncrement[x];
    uint xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint rot = oldstate >> 59u;
    uint pcg = (xorshifted >> rot) | (xorshifted << ((-rot) & 31));

    return pcg;
}

__kernel void seedNoise(
    global ulong *randomState,
    global ulong *randomIncrement,
    global ulong *initState,
    global ulong *initSeq
) {
    const int x = get_global_id(0);

    randomState[x] = 0U;
    randomIncrement[x] = (initSeq[x] << 1u) | 1u;
    pcg32Random(randomState, randomIncrement, x);
    randomState[x] += initState[x];
    pcg32Random(randomState, randomIncrement, x);
}

inline float uniformRand(
    global ulong *randomState,
    global ulong *randomIncrement,
    int x
) {
    return (float)pcg32Random(randomState, randomIncrement, x) / PCG_MAX_1;
}

inline float gaussianRand(
    global ulong *randomState,
    global ulong *randomIncrement,
    int x
) {
    return inverseNormalCdf(uniformRand(randomState, randomIncrement, x));
}

/**
 * Particle stuff
 */

typedef struct Particle {
    float x, y;
    float phi;
    float velocity;
} Particle;

inline float clip(float in, float lower, float upper) {
    if (in < lower) {
        return lower;
    }
    if (in > upper) {
        return upper;
    }

    return in;
}

__kernel void setParticleVels(
    global Particle *particles,
    global ulong *randomState,
    global ulong *randomIncrement,
    float velocitySpread,
    float baseVelocity
) {
    const int x = get_global_id(0);
    particles[x].velocity = uniformRand(randomState, randomIncrement, x) * velocitySpread + baseVelocity;
}

__kernel void initParticles(
    global Particle *particles,
    global ulong *randomState,
    global ulong *randomIncrement,
    float velocitySpread,
    float baseVelocity,
    int size_x,
    int size_y
) {
    const int x = get_global_id(0);

    // Squaretangle
    // for (i = 0; i < particlesPerThread; i++) {
    //     particle = particles[thread][i];

    //     particle.x = clip(0.5 + RANDN() * 0.31999, 0.3, 0.7) * size_x;
    //     particle.y = clip(0.5 + RANDN() * 0.31999, 0.3, 0.7) * size_y;
    //     particle.phi = 2 * M_PI * UNI();

    //     particles[thread][i] = particle;
    // }

    // Circle
    float xc = size_x * 0.5;
    float yc = size_y * 0.5;

    float theta = uniformRand(randomState, randomIncrement, x) * 2 * M_PI;
    float rad = (gaussianRand(randomState, randomIncrement, x) / 32. + 0.25);

    particles[x].x = clip(cos(theta) * rad * size_y + xc, 0., size_x);
    particles[x].y = clip(sin(theta) * rad * size_y + yc, 0., size_y);
    particles[x].phi = atan2(yc - particles[x].y, xc - particles[x].x);

    setParticleVels(particles, randomState, randomIncrement, velocitySpread, baseVelocity);
}

__kernel void diffuse(global float *input, global float *output, float one_9)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    
    const int W = get_global_size(0);
    const int H = get_global_size(1);
    
    int index;
    
    index = (W * y + x);
    
    int k, l, km, lm;
    float conv = 0;

    for (k = -1; k < 2; k++) {
        km = x + k;

        if (km == -1) {
#ifdef PERIODIC_BOUNDARY
            km = W-1;
#else
            // km = 0;
            continue;
#endif
        } else if (km == W) {
#ifdef PERIODIC_BOUNDARY
            km = 0;
#else
            // km = W-1;
            continue;
#endif
        }
        for (l = -1; l < 2; l++) {
            lm = y + l;
            
            if (lm == -1) {
#ifdef PERIODIC_BOUNDARY
                lm = H-1;
#else
                // lm = 0
                continue;
#endif
            } else if (lm == H) {
#ifdef PERIODIC_BOUNDARY
                lm = 0;
#else
                // lm = H-1;
                continue;
#endif
            }
            
            conv += input[km + W * lm];
        }
    }

    
    output[index] = conv * one_9;
}

__kernel void moveParticles(
    global Particle *particles,
    global float *trail,
    global ulong *randomState,
    global ulong *randomIncrement,
    int size_x,
    int size_y,
    float sensorAngle,
    float sensorDist,
    float rotationAngle
) {
    const int x = get_global_id(0);
    const int nParticles = get_global_size(0);
    
    float fl, fc, fr;
    int flx, fly, fcx, fcy, frx, fry;

    Particle particle = particles[x];

    flx = particle.x + cos(particle.phi - sensorAngle) * sensorDist;
    fly = particle.y + sin(particle.phi - sensorAngle) * sensorDist;

    fcx = particle.x + cos(particle.phi) * sensorDist;
    fcy = particle.y + sin(particle.phi) * sensorDist;

    frx = particle.x + cos(particle.phi + sensorAngle) * sensorDist;
    fry = particle.y + sin(particle.phi + sensorAngle) * sensorDist;

#ifdef PERIODIC_BOUNDARY
    fl = trail[clamp((int)flx, 0, size_x - 1) + size_x * clamp((int)fly, 0, size_y - 1)];
    fc = trail[clamp((int)fcx, 0, size_x - 1) + size_x * clamp((int)fcy, 0, size_y - 1)];
    fr = trail[clamp((int)frx, 0, size_x - 1) + size_x * clamp((int)fry, 0, size_y - 1)];
#else
    fl = trail[(((int)flx + size_x) % size_x) + size_x * (((int)fly + size_y) % size_y)];
    fc = trail[(((int)fcx + size_x) % size_x) + size_x * (((int)fcy + size_y) % size_y)];
    fr = trail[(((int)frx + size_x) % size_x) + size_x * (((int)fry + size_y) % size_y)];
#endif

    if (fc < fl && fc < fr) {
        // particle.phi += rotationAngle * (uniformRand(randomState, randomIncrement, x) > 0.5 ? 1 : -1);
        particle.phi += rotationAngle * (uniformRand(randomState, randomIncrement, x) * 2 - 1);
    }
    else if (fl > fc && fc > fr) {
        particle.phi -= rotationAngle;
    }
    else if (fl < fc && fc < fr) {
        particle.phi += rotationAngle;
    }

    particle.x += cos(particle.phi) * particle.velocity;
    particle.y += sin(particle.phi) * particle.velocity;

#ifdef PERIODIC_BOUNDARY
    if (particle.x < 0) {
        particle.x += size_x;
    }
    else if (particle.x >= size_x) {
        particle.x -= size_x;
    }

    if (particle.y < 0) {
        particle.y += size_y;
    }
    else if (particle.y >= size_y) {
        particle.y -= size_y;
    }
#else
    if (particle.x < 0) {
        particle.x *= -1;
        particle.phi = M_PI_F - particle.phi;
        // particle.phi = 0;
    }
    else if (particle.x >= size_x) {
        particle.x = 2 * size_x - particle.x;
        particle.phi = M_PI_F - particle.phi;
        // particle.phi = M_PI_F;
    }

    if (particle.y < 0) {
        particle.y *= -1;
        particle.phi *= -1;
        // particle.phi = M_PI_2_F;
    }
    else if (particle.y >= size_y) {
        particle.y = 2 * size_y - particle.y;
        particle.phi *= -1;
        // particle.phi = -M_PI_2_F;
    }
#endif

    particles[x] = particle;
}

__kernel void depositStuff(
    global Particle *particles, 
    global float *trail, 
    int size_x, 
    int size_y,
    float depositAmount
) {
    const int x = get_global_id(0);

    Particle particle = particles[x];

    trail[(int)particle.x + size_x * ((int)particle.y)] += depositAmount;
}

inline float sigmoid(float x) {
    return 1. / (1. + exp(-x));
}

inline float rescaleTrail(float x) {
    if (x < 0.7) {
        return sqrt(x);
    }

    return sigmoid(x) * rescaleFactor;
}

inline float rescaleTrail2(float x) {
    if (x < 0.7) {
        return sqrt(x) * invsqrt07;
    }

    return 0.5 + cos(sqrt(x - 0.7)) * 0.5;
}

__kernel void resetTrail(
    global float *trail1,
    global float *trail2
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    
    const int W = get_global_size(0);

    int ind = (x + W * y);
    
    trail1[ind] = 0;
    trail2[ind] = 0;
}

__kernel void processTrail(
    global float *trail, 
    global uint *image, 
    global unsigned int *colourMap,
    int nColours
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    
    const int W = get_global_size(0);
    const int H = get_global_size(1);

    int ind, ind2;

    ind = (x + W * y);

    float brightness = rescaleTrail2(trail[ind]);
    ind2 = 3 * (int)(fmin(imageLim, brightness) * nColours);

    for (int k = 0; k < 3; k++) {
        image[3 * ind + k] = colourMap[ind2 + k];
    }
}

__kernel void resetImage(
    global uint *image
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    
    const int W = get_global_size(0);
    const int H = get_global_size(1);

    int ind = (x + W * y);

    for (int k = 0; k < 3; k++) {
        image[3 * ind + k] = 4294967295;
    }
}

__kernel void renderParticles(
    global Particle *particles, 
    global uint *image,
    int W
) {
    const int x = get_global_id(0);
    Particle particle = particles[x];
    int ind = ((int)particle.x + W * (int)particle.y);

    for (int k = 0; k < 3; k++) {
        image[3 * ind + k] *= 0.8;
    }
}

__kernel void invertImage(
    global uint *image
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    
    const int W = get_global_size(0);
    const int H = get_global_size(1);

    int ind = (x + W * y);

    for (int k = 0; k < 3; k++) {
        image[3 * ind + k] = sqrt((4294967295 - image[3 * ind + k]) / (float)4294967295) * 4294967295;
    }
}

__kernel void lagImage(
    global uint *image,
    global uint *image2
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    
    const int W = get_global_size(0);
    const int H = get_global_size(1);

    int ind = (x + W * y);

    for (int k = 0; k < 3; k++) {
        image2[3 * ind + k] = 0.85 * image2[3 * ind + k] + 0.15 * image[3 * ind + k];
    }
}