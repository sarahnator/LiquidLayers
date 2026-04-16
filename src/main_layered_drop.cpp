#include "imgui.h"
#include "polyscope/curve_network.h"
#include "polyscope/polyscope.h"
#include "simulation_layered_drop.h"
#include <algorithm>
#include <string>

static LayeredDropSimulation *g_sim = nullptr;
static bool g_running = false;
static int g_spf = 5;

static const char *kModelNames[] = {"NewtonianFluid (WCSPH/MPM)",
                                    "FixedCorotated", "DruckerPrager"};

static void applyDefaultLayeredPreset(LayeredDropSimulation &sim) {
  auto &scene = sim.sceneParamsMutable();
  auto &water = sim.materialParamsMutable(MaterialType::Water);
  auto &soil = sim.materialParamsMutable(MaterialType::Soil);
  auto &sand = sim.materialParamsMutable(MaterialType::Sand);
  auto &rock = sim.materialParamsMutable(MaterialType::Rock);

  scene.layer_gap = 0.04f;

  water = defaultMaterialParams(MaterialType::Water);
  soil = defaultMaterialParams(MaterialType::Soil);
  sand = defaultMaterialParams(MaterialType::Sand);
  rock = defaultMaterialParams(MaterialType::Rock);

  sim.toggles.bc_top = false;
  sim.toggles.enable_viscosity = true;
  sim.toggles.model_water_tait = true;
}

static void applySplashSpillPreset(LayeredDropSimulation &sim) {
  auto &scene = sim.sceneParamsMutable();
  auto &water = sim.materialParamsMutable(MaterialType::Water);
  auto &soil = sim.materialParamsMutable(MaterialType::Soil);
  auto &sand = sim.materialParamsMutable(MaterialType::Sand);
  auto &rock = sim.materialParamsMutable(MaterialType::Rock);

  // Open the top so splash can rise, and separate the layers slightly so
  // fluid / granular rearrangement is easier to observe.
  sim.toggles.bc_top = false;
  scene.layer_gap = 0.06f;

  water.model = ConstitutiveModel::WeaklyCompressibleFluid;
  water.bulk_modulus = 220.f;
  water.gamma = 6.f;
  water.viscosity = 0.01f;
  water.computeDerived();

  // Make soil spill more readily by using the same Drucker-Prager family as
  // sand, but a bit softer and slightly more cohesive.
  soil.model = ConstitutiveModel::DruckerPrager;
  soil.youngs_modulus = 4.0e3f;
  soil.poisson_ratio = 0.28f;
  soil.friction_angle = 18.f;
  soil.cohesion = 2.f;
  soil.computeDerived();

  sand.model = ConstitutiveModel::DruckerPrager;
  sand.youngs_modulus = 6.0e3f;
  sand.poisson_ratio = 0.26f;
  sand.friction_angle = 16.f;
  sand.cohesion = 0.f;
  sand.computeDerived();

  rock.model = ConstitutiveModel::FixedCorotated;
  rock.youngs_modulus = 3.0e4f;
  rock.poisson_ratio = 0.22f;
  rock.computeDerived();
}

static void applyDensityStratifiedFluidPreset(LayeredDropSimulation &sim) {
  auto &scene = sim.sceneParamsMutable();
  auto &water = sim.materialParamsMutable(MaterialType::Water);
  auto &soil = sim.materialParamsMutable(MaterialType::Soil);
  auto &sand = sim.materialParamsMutable(MaterialType::Sand);
  auto &rock = sim.materialParamsMutable(MaterialType::Rock);

  // Use the same weakly compressible Newtonian fluid model for every layer so
  // the dominant contrast is density. We intentionally choose an unstable
  // density ordering (heavy fluid on top of light fluid) so the rearrangement
  // is easy to see after impact and during sloshing.
  sim.toggles.bc_top = false;
  sim.toggles.enable_viscosity = true;
  sim.toggles.model_water_tait = true;
  scene.layer_gap = 0.03f;

  for (auto *mp : {&water, &soil, &sand, &rock}) {
    mp->model = ConstitutiveModel::WeaklyCompressibleFluid;
    mp->bulk_modulus = 260.f;
    mp->gamma = 7.f;
    mp->viscosity = 0.02f;
    mp->computeDerived();
  }

  // Bottom -> top labels remain rock / sand / soil / water, but here they are
  // all fluids with different densities.
  rock.density0 = 900.f;
  sand.density0 = 1100.f;
  soil.density0 = 1500.f;
  water.density0 = 2200.f;

  rock.computeDerived();
  sand.computeDerived();
  soil.computeDerived();
  water.computeDerived();
}

static void refreshAfterReinitialize(LayeredDropSimulation &sim) {
  // Presets can change the particle count (for instance by adding layer gaps),
  // so we rebuild the Polyscope point cloud rather than assuming the old cloud
  // still has the right size.
  polyscope::removeStructure("layered_drop_particles", false);
  sim.registerPolyscope();
}

static void registerBoxBoundary(double xmin, double xmax, double ymin,
                                double ymax) {
  Eigen::MatrixXd nodes(4, 3);
  nodes << xmin, ymin, 0, xmax, ymin, 0, xmax, ymax, 0, xmin, ymax, 0;

  Eigen::MatrixXi edges(4, 2);
  edges << 0, 1, 1, 2, 2, 3, 3, 0;

  auto *box = polyscope::registerCurveNetwork("box boundary", nodes, edges);
  box->setRadius(0.002);
  box->setColor({0.0, 0.0, 0.0});
}

static void showParticleInfo(int idx) {
  if (idx < 0 || idx >= static_cast<int>(g_sim->particles().size()))
    return;
  const auto &p = g_sim->particles()[idx];
  const auto &mp = g_sim->materialParams(p.material);

  const char *names[] = {"Water", "Soil", "Sand", "Rock"};
  ImGui::Text("material : %s", names[(int)p.material]);
  ImGui::Text("pos      : (%.3f, %.3f)", p.pos.x(), p.pos.y());
  ImGui::Text("vel      : (%.3f, %.3f)", p.vel.x(), p.vel.y());
  ImGui::Text("J=det(F) : %.4f", p.F.determinant());
  ImGui::Text("F        : [[%.2f %.2f][%.2f %.2f]]", p.F(0, 0), p.F(0, 1),
              p.F(1, 0), p.F(1, 1));
  ImGui::Separator();
  ImGui::Text("rho      : %.1f", mp.density0);
  ImGui::Text("E        : %.3e", mp.youngs_modulus);
  ImGui::Text("nu       : %.3f", mp.poisson_ratio);
  ImGui::Text("bulk     : %.3e", mp.bulk_modulus);
  ImGui::Text("gamma    : %.3f", mp.gamma);
  ImGui::Text("visc     : %.3f", mp.viscosity);
  ImGui::Text("phi      : %.1f", mp.friction_angle);
}

static void showMaterialEditor(const char *label, MaterialParams &mp) {
  if (!ImGui::TreeNode(label))
    return;

  ImGui::PushID(label);

  int model = static_cast<int>(mp.model);
  ImGui::Combo("Constitutive model##model", &model, kModelNames,
               IM_ARRAYSIZE(kModelNames));
  mp.model = static_cast<ConstitutiveModel>(model);

  ImGui::InputFloat("Density##density", &mp.density0);
  ImGui::InputFloat("Young's modulus##E", &mp.youngs_modulus, 0.f, 0.f, "%.3e");
  ImGui::SliderFloat("Poisson ratio##nu", &mp.poisson_ratio, 0.0f, 0.45f);
  ImGui::InputFloat("Bulk modulus##K", &mp.bulk_modulus, 0.f, 0.f, "%.3e");
  ImGui::SliderFloat("Gamma##gamma", &mp.gamma, 1.0f, 12.0f);
  ImGui::SliderFloat("Viscosity##muv", &mp.viscosity, 0.0f, 2.0f);
  ImGui::SliderFloat("Friction angle##phi", &mp.friction_angle, 0.0f, 60.0f);
  ImGui::InputFloat("Cohesion##cohesion", &mp.cohesion);

  mp.density0 = std::max(mp.density0, 1.f);
  mp.youngs_modulus = std::max(mp.youngs_modulus, 1.f);
  mp.computeDerived();

  ImGui::Text("mu        = %.3e", mp.mu);
  ImGui::Text("lambda    = %.3e", mp.lambda_lame);
  ImGui::Text("alpha_dp  = %.3e", mp.alpha_dp);
  ImGui::PopID();
  ImGui::TreePop();
}

void uiCallback() {
  auto &t = g_sim->toggles;
  auto &p = g_sim->paramsMutable();
  auto &scene = g_sim->sceneParamsMutable();

  ImGui::SetNextWindowPos(ImVec2(320, 10), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(380, 800), ImGuiCond_FirstUseEver);
  ImGui::Begin("Layered drop MPM debug");

  ImGui::Text("Particles : %zu", g_sim->particles().size());
  ImGui::Text("Frame     : %d", g_sim->frameCount());
  ImGui::Text("dt        : %.2e", p.dt);
  ImGui::Separator();

  if (ImGui::Button(g_running ? "Pause" : "Play"))
    g_running = !g_running;
  ImGui::SameLine();
  if (ImGui::Button("Step x1")) {
    g_sim->step();
    g_sim->updatePolyscope();
  }
  ImGui::SameLine();
  if (ImGui::Button("Step x20")) {
    for (int i = 0; i < 20; ++i)
      g_sim->step();
    g_sim->updatePolyscope();
  }
  ImGui::SliderInt("Steps / frame", &g_spf, 1, 30);
  if (ImGui::Button("Reset simulation")) {
    g_running = false;
    p.computeDerived();
    g_sim->initialize();
    refreshAfterReinitialize(*g_sim);
  }
  if (ImGui::Button("Apply default layered preset")) {
    applyDefaultLayeredPreset(*g_sim);
    p.computeDerived();
    g_sim->initialize();
    refreshAfterReinitialize(*g_sim);
    g_running = false;
  }
  if (ImGui::Button("Apply splash / spill preset")) {
    applySplashSpillPreset(*g_sim);
    p.computeDerived();
    g_sim->initialize();
    refreshAfterReinitialize(*g_sim);
    g_running = false;
  }
  if (ImGui::Button("Apply density-stratified fluid preset")) {
    applyDensityStratifiedFluidPreset(*g_sim);
    p.computeDerived();
    g_sim->initialize();
    refreshAfterReinitialize(*g_sim);
    g_running = false;
  }

  ImGui::Separator();
  ImGui::TextWrapped(
      "This scene keeps the simple falling block setup, but divides the block "
      "into rock / sand / soil / water layers so you can debug how the four "
      "constitutive models interact during impact. The density-stratified "
      "fluid preset converts all four layers into weakly compressible "
      "Newtonian fluids with different densities so you can watch density-"
      "driven inversion and rearrangement.");

  ImGui::Separator();
  ImGui::Checkbox("Enable gravity", &t.enable_gravity);
  ImGui::Checkbox("Enable stress", &t.enable_stress);
  ImGui::Checkbox("Enable water viscosity", &t.enable_viscosity);
  ImGui::Checkbox("Fluid EOS / pressure", &t.model_water_tait);
  ImGui::TextWrapped(
      "The constitutive-model dropdowns below now control each layer directly. "
      "These two fluid toggles apply to any layer currently using the "
      "Newtonian "
      "fluid model.");

  if (ImGui::CollapsingHeader("Boundary conditions",
                              ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::Checkbox("Left wall", &t.bc_left);
    ImGui::SameLine(150.f);
    ImGui::Checkbox("Right wall", &t.bc_right);
    ImGui::Checkbox("Bottom wall", &t.bc_bottom);
    ImGui::SameLine(150.f);
    ImGui::Checkbox("Top wall", &t.bc_top);
  }

  if (ImGui::CollapsingHeader("Simulation parameters",
                              ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::InputFloat("dt", &p.dt, 0.f, 0.f, "%.2e");
    p.dt = std::max(p.dt, 1e-7f);
    ImGui::InputFloat("gravity", &p.gravity);
    ImGui::InputFloat("block width", &scene.block_w);
    ImGui::InputFloat("block height", &scene.block_h);
    ImGui::InputFloat("block center x", &scene.block_center_x);
    ImGui::InputFloat("block center y", &scene.block_center_y);
    ImGui::SliderFloat("layer gap", &scene.layer_gap, 0.0f, 0.15f);
    scene.block_w = std::max(scene.block_w, 0.1f);
    scene.block_h = std::max(scene.block_h, 0.1f);
    scene.layer_gap = std::max(scene.layer_gap, 0.0f);
  }

  if (ImGui::CollapsingHeader("Material parameters",
                              ImGuiTreeNodeFlags_DefaultOpen)) {
    showMaterialEditor("Water",
                       g_sim->materialParamsMutable(MaterialType::Water));
    showMaterialEditor("Soil",
                       g_sim->materialParamsMutable(MaterialType::Soil));
    showMaterialEditor("Sand",
                       g_sim->materialParamsMutable(MaterialType::Sand));
    showMaterialEditor("Rock",
                       g_sim->materialParamsMutable(MaterialType::Rock));
    ImGui::TextWrapped(
        "Density changes affect newly initialized particle masses. Reset after "
        "large density edits so the block is re-seeded consistently. The "
        "constitutive-model dropdown lets you swap a layer between a weakly "
        "compressible Newtonian fluid, an elastic solid, and Drucker-Prager "
        "plastic behavior from the UI.");
  }

  if (ImGui::CollapsingHeader("Particle inspector")) {
    static int inspect_idx = 0;
    ImGui::InputInt("Index", &inspect_idx);
    inspect_idx = std::clamp(inspect_idx, 0,
                             std::max(0, (int)g_sim->particles().size() - 1));
    showParticleInfo(inspect_idx);
  }

  if (ImGui::CollapsingHeader("Expected behavior guide")) {
    ImGui::TextWrapped(
        "Rock should be the stiffest layer. Sand should yield and flow when "
        "plasticity is enabled. Soil can be either elastic or yielding, "
        "depending on the chosen constitutive model. Water should slosh and "
        "resist compression without carrying sustained shear like the solids. "
        "Opening the top wall and inserting small layer gaps makes splash and "
        "spill behavior easier to observe. Turning stress OFF should make the "
        "whole block lose coherence and behave like falling particles.");
  }

  ImGui::End();

  if (g_running) {
    for (int i = 0; i < g_spf; ++i)
      g_sim->step();
    g_sim->updatePolyscope();
  }
}

int main() {
  SimParams params;
  params.domain_w = 10.f;
  params.domain_h = 6.f;
  params.ppc = 4;
  params.grid_nx = 80;
  params.grid_ny = 48;
  params.dt = 5e-5f;
  params.gravity = -9.8f;
  params.computeDerived();

  LayeredDropSimulation sim(params);
  applySplashSpillPreset(sim);
  sim.initialize();
  g_sim = &sim;

  polyscope::init();
  registerBoxBoundary(0.f, params.domain_w, 0.f, params.domain_h);

  polyscope::options::programName = "Layered drop MPM debug";
  polyscope::view::upDir = polyscope::UpDir::YUp;
  polyscope::view::style = polyscope::NavigateStyle::Planar;
  polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::None;

  sim.registerPolyscope();
  polyscope::view::lookAt({5., 3., 20.}, {5., 3., 0.}, {0., 1., 0.});
  polyscope::state::userCallback = uiCallback;
  polyscope::show();
  return 0;
}
