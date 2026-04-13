# MPM Fluid Simulation
A C++ implementation of the Material Point Method for multi-layer fluid simulation,
inspired by Grant Kot's [Liquid Layers](https://grantkot.com/ll/).

## Build

```bash
# Clone / enter project
cd mpm_fluid

# Configure (downloads Polyscope + Eigen automatically via CMake FetchContent)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Compile
cmake --build build -j$(nproc)

# Run
./build/mpm_fluid
```

**Requirements**: CMake ≥ 3.18, a C++17 compiler (GCC 9+, Clang 10+, MSVC 2019+),
and OpenGL 3.3+ (any modern desktop GPU).

First build takes ~3-5 minutes — Polyscope is fetched and compiled from source.

---

## Project phases

| Phase | What you implement | Key concepts |
|-------|--------------------|--------------|
| **1** | Boilerplate + colored particle layers | Polyscope setup, particle struct |
| **2** | Particle & grid data structures | B-spline weight functions, P-G indexing |
| **3** | P2G + G2P transfer loop | APIC, mass/momentum rasterization |
| **4** | Constitutive model | Deformation gradient F, stress tensor, equation of state |
| **5** | Multi-material + boundaries | Per-particle material tag, sticky walls |
| **6** | Polish + interaction | Mouse forces, surface tension, UI sliders |

---

## File structure

```
mpm_fluid/
├── CMakeLists.txt          # Build config; fetches Polyscope + Eigen
└── src/
    ├── types.h             # Particle, GridCell, MaterialType, SimParams
    ├── simulation.h        # Simulation class declaration
    ├── simulation.cpp      # initialize(), registerPolyscope(), updatePolyscope()
    └── main.cpp            # Polyscope init, render loop, ImGui UI panel
```

---

## Key MPM resources

- **Jiang et al. 2016** — "The Material Point Method for Simulating Continuum Materials"
  SIGGRAPH course notes (free PDF, search "MPM course SIGGRAPH 2016")
- **Hu et al. 2018** — "A Moving Least Squares Material Point Method with Displacement
  Discontinuity and Two-Way Rigid Body Coupling" (APIC/MLS-MPM)
- **Polyscope docs** — https://polyscope.run/
- **Grant Kot's source** — study it *after* you've implemented your own version!

---

## Phase 2 

The MPM timestep loop looks like this:

```
for each timestep:
  1. Reset grid  (zero mass and momentum on every node)
  2. P2G         (each particle spreads its mass + momentum to nearby grid nodes
                  using B-spline weights; also compute volume if first step)
  3. Grid forces (compute stress → nodal forces from constitutive model)
  4. Grid update (F = ma on each node; apply gravity; enforce boundaries)
  5. G2P         (each particle reads velocity from nearby grid nodes;
                  update particle velocity, APIC affine matrix C, then position)
```

The B-spline weight for a particle at `xp` affecting grid node `i` is:
```
w = N((xp - xi) / dx)
where N(x) = 0.75 - x^2          for |x| < 0.5
             0.5*(1.5 - |x|)^2   for 0.5 <= |x| < 1.5
             0                   otherwise
```
This is the quadratic B-spline — it affects a 3×3 stencil of grid nodes per particle.
