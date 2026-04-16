#pragma once

#include "types.h"
#include <array>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
//  SimTogglesLayeredDrop
//
//  Runtime feature flags for the layered-drop debugging scene. This keeps the
//  same constitutive model toggles as the full MPM debug scene, but simplifies
//  the initial condition to a single falling multilayer block in a box.
// ─────────────────────────────────────────────────────────────────────────────
struct SimTogglesLayeredDrop {
  bool render_particles = true;

  bool enable_gravity = true;
  bool enable_stress = true;
  bool enable_viscosity = true;

  bool model_water_tait = true;
  bool model_soil_elastic = true;
  bool model_sand_plastic = true;
  bool model_rock_elastic = true;

  bool bc_left = true;
  bool bc_right = true;
  bool bc_bottom = true;
  bool bc_top = true;
};

// ─────────────────────────────────────────────────────────────────────────────
//  LayeredDropSceneParams
//
//  Extra scene parameters beyond the generic MPM SimParams. These control the
//  position and dimensions of the dropped multilayer block.
// ─────────────────────────────────────────────────────────────────────────────
struct LayeredDropSceneParams {
  float block_w = 1.2f;
  float block_h = 1.6f;
  float block_center_x = 5.0f;
  float block_center_y = 4.7f;

  // Optional empty spacing inserted between the four material bands inside the
  // dropped block. A small nonzero gap makes rearrangement and splash / spill
  // behavior easier to observe because the layers do not begin in perfectly
  // bonded contact.
  float layer_gap = 0.0f;
};

class LayeredDropSimulation {
public:
  explicit LayeredDropSimulation(SimParams params = {});

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

  const LayeredDropSceneParams &sceneParams() const { return scene_params_; }
  LayeredDropSceneParams &sceneParamsMutable() { return scene_params_; }

  const MaterialParams &materialParams(MaterialType m) const;
  MaterialParams &materialParamsMutable(MaterialType m);

  int frameCount() const { return frame_; }

  SimTogglesLayeredDrop toggles;

private:
  SimParams params_;
  LayeredDropSceneParams scene_params_;
  std::vector<Particle> particles_;
  std::vector<GridNode> grid_;
  int frame_ = 0;

  MaterialParams water_params_ = defaultMaterialParams(MaterialType::Water);
  MaterialParams soil_params_ = defaultMaterialParams(MaterialType::Soil);
  MaterialParams sand_params_ = defaultMaterialParams(MaterialType::Sand);
  MaterialParams rock_params_ = defaultMaterialParams(MaterialType::Rock);

  static constexpr const char *kCloudName = "layered_drop_particles";

  void buildRenderArrays(std::vector<std::array<double, 3>> &,
                         std::vector<std::array<double, 3>> &) const;
  void addLayeredBlock(float x_min, float x_max, float y_min, float y_max);

  Eigen::Matrix2f kirchhoffStress(const Particle &p) const;
  Eigen::Matrix2f stressFluid(const Particle &p,
                              const MaterialParams &mp) const;
  Eigen::Matrix2f stressFixedCorotated(const Particle &p,
                                       const MaterialParams &mp) const;
  Eigen::Matrix2f stressDruckerPrager(const Particle &p,
                                      const MaterialParams &mp) const;
  void projectDruckerPrager(Particle &p, const MaterialParams &mp) const;
};
