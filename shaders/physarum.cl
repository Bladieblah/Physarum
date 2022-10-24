__constant sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_NEAREST | CLK_ADDRESS_NONE;

__constant float rescaleFactor = 1.25213309998062819936804151;
__constant float invsqrt07 = 1.19522860933439363996881717;
__constant float imageLim = 0.999;

typedef struct Particle {
    float x, y;
    float phi;
    float velocity;
} Particle;

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
            km = W-1;
        } else if (km == W) {
            km = 0;
        }
        for (l = -1; l < 2; l++) {
            lm = y + l;
            
            if (lm == -1) {
                lm = H-1;
            } else if (lm == H) {
                lm = 0;
            }
            
            conv += input[km + W * lm];
        }
    }

    
    output[index] = conv * one_9;
}

__kernel void moveParticles(
    global Particle *particles, 
    global float *trail, 
    global float *random, 
    int size_x, 
    int size_y,
    float sensorAngle,
    float sensorDist,
    float rotationAngle
) {
	const int x = get_global_id(0);
	const int nParticles = get_global_size(0);
	
    float fl, fc, fr;
    float flx, fly, fcx, fcy, frx, fry;

    Particle particle = particles[x];

    // if (random[x] < 0.99) {
    flx = particle.x + cos(particle.phi - sensorAngle) * sensorDist;
    fly = particle.y + sin(particle.phi - sensorAngle) * sensorDist;

    fcx = particle.x + cos(particle.phi) * sensorDist;
    fcy = particle.y + sin(particle.phi) * sensorDist;

    frx = particle.x + cos(particle.phi + sensorAngle) * sensorDist;
    fry = particle.y + sin(particle.phi + sensorAngle) * sensorDist;

    fl = trail[(int)flx + size_x * (int)fly];
    fc = trail[(int)fcx + size_x * (int)fcy];
    fr = trail[(int)frx + size_x * (int)fry];

    if (fc < fl && fc < fr) {
        particle.phi += rotationAngle * (random[x+1] > 0.5 ? 1 : -1);
    }
    else if (fl > fc && fc > fr) {
        particle.phi -= rotationAngle;
    }
    else if (fl < fc && fc < fr) {
        particle.phi += rotationAngle;
    }
    // }
    // else {
    //     particle.phi += 10 * rotationAngle * (random[x+2] > 0.5 ? 1 : -1);
    // }

    particle.x += cos(particle.phi) * particle.velocity;
    particle.y += sin(particle.phi) * particle.velocity;

    // Periodic boundary
    /*
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
    //*/

    // Hard boundary
    // /*
    if (particle.x < 0) {
        particle.x *= -1;
        // particle.phi = M_PI_2_F - particle.phi;
        particle.phi = 0;
    }
    else if (particle.x >= size_x) {
        particle.x = 2 * size_x - particle.x;
        // particle.phi = M_PI_2_F - particle.phi;
        particle.phi = M_PI_F;
    }

    if (particle.y < 0) {
        particle.y *= -1;
        // particle.phi *= -1;
        particle.phi = M_PI_2_F;
    }
    else if (particle.y >= size_y) {
        particle.y = 2 * size_y - particle.y;
        // particle.phi *= -1;
        particle.phi = -M_PI_2_F;
    }
    //*/

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
	const int nParticles = get_global_size(0);

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

__kernel void processTrail(
    global float *trail, 
    global uint *image, 
    global float *colourMap,
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
        image[3 * ind + k] = colourMap[ind2 + k] * 4294967295;
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