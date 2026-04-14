#pragma once

#include "types_elastic_drop.h"
#include <array>
#include <vector>

struct SimToggles {
  bool render_particles = true;
  bool enable_gravity = true;
  bool enable_stress = true;

  bool bc_left = true;
  bool bc_right = true;
  bool bc_bottom = true;
  bool bc_top = true;

  // Sticky wall: zero the inward component. If false, wall contact is only
  // enforced by position clamping during advection.
  bool sticky_walls = true;
};

class Simulation {
public:
  explicit Simulation(SimParams params = {});

  void initialize();
  void step();

  void clearGrid();
  void substep_P2G();
  void substep_gridUpdate();
  void substep_G2P();
  void substep_advect();

  int gridIdx(int i, int j) const { return j * params_.grid_nx + i; }
  bool inGrid(int i, int j) const {
    return i >= 0 && i < params_.grid_nx && j >= 0 && j < params_.grid_ny;
  }

  void registerPolyscope();
  void updatePolyscope();

  const std::vector<Particle> &particles() const { return particles_; }
  const SimParams &params() const { return params_; }
  SimParams &paramsMutable() { return params_; }
  int frameCount() const { return frame_; }

  SimToggles toggles;

private:
  SimParams params_;
  std::vector<Particle> particles_;
  std::vector<GridNode> grid_;
  int frame_ = 0;

  static constexpr const char *kCloudName = "elastic_drop_particles";

  void buildRenderArrays(std::vector<std::array<double, 3>> &,
                         std::vector<std::array<double, 3>> &) const;
  void addElasticBlock(float x_min, float x_max, float y_min, float y_max);
  void enforceParticleBounds(Particle &p) const;

  Eigen::Matrix2f kirchhoffStress(const Particle &p) const;
  Eigen::Matrix2f stressLinearElastic(const Particle &p,
                                      const MaterialParams &mp) const;
};
