# Physarum simulation

Implementation of [Sage Jenson's Physarum simulation](https://cargocollective.com/sagejenson/physarum), built with OpenCL and ImGUI.

## Installing
Has only been tested on MacOS, both AMD and M1. Run it with

```bash
make
./physarum.out
```

## Controls
You can control the simulation with the sliders in the `Parameters` subwindow
- Sensor angle: Angle between the sensors and the heading of the particle.
- Sensor dist: Distance between the sensors and the particle center.
- Rot angle: How much the particle rotates.
- Velocity: The particles have random velocities, uniformly distrubuted in the interval [0.1, 0.1 + velocity].
- Amount: How much a particle deposits on the trail each frame. Higher amounts mean shorter decay times.
- Avg: The stable average on the trail. Higher amounts mean longer decay times.
### Hotkeys
- t: Switch between trail and particle render modes.
- u: Randomise parameters.
- r: Restart the simulation with the current parameters.
- q: Quit the simulation

## Recording
Record a simulation by running with the `-s` option, requires ffmpeg to work.

```bash
make
./physarum.out -s
```

## Gallery
![Demo 1](public/demo1.gif)
