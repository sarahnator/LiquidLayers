#include "imgui.h"
#include "polyscope/curve_network.h"
#include "polyscope/polyscope.h"
#include "simulation_two_layer_drop.h"
#include <algorithm>
#include <string>

static TwoLayerDropSimulation *g_sim = nullptr;
static bool g_running = false;
static int g_spf = 5;

static const char *kModelNames[] = {"NewtonianFluid (WCSPH/MPM)",
                                    "FixedCorotated", "DruckerPrager"};
static const char *kMaterialNames[] = {"Water", "Soil", "Sand", "Rock"};

static void applyDefaultTwoLayerPreset(TwoLayerDropSimulation &sim) {
  auto &scene = sim.sceneParamsMutable();
  auto &bottom = sim.materialParamsMutable(MaterialType::Rock);
  auto &top = sim.materialParamsMutable(MaterialType::Water);

  scene.block_w = 0.75f;
  scene.block_h = 3.2f;
  scene.block_center_x = 5.0f;
  scene.block_center_y = 4.1f;
  scene.layer_gap = 0.02f;
  scene.buoyancy_strength = 1.25f;
  scene.phase_separation_strength = 2.5f;
  scene.bottom_material = MaterialType::Rock;
  scene.top_material = MaterialType::Water;

  bottom = defaultMaterialParams(MaterialType::Rock);
  top = defaultMaterialParams(MaterialType::Water);

  sim.toggles.enable_gravity = true;
  sim.toggles.enable_stress = true;
  sim.toggles.enable_viscosity = true;
  sim.toggles.enable_buoyancy_helper = true;
  sim.toggles.enable_phase_separation_helper = true;
  sim.toggles.model_water_tait = true;
  sim.toggles.bc_left = true;
  sim.toggles.bc_right = true;
  sim.toggles.bc_bottom = true;
  sim.toggles.bc_top = false;
}

static void applyTwoLayerDensityInversionPreset(TwoLayerDropSimulation &sim) {
  auto &scene = sim.sceneParamsMutable();
  auto &bottom = sim.materialParamsMutable(MaterialType::Rock);
  auto &top = sim.materialParamsMutable(MaterialType::Water);

  scene.block_w = 0.75f;
  scene.block_h = 3.2f;
  scene.block_center_x = 5.0f;
  scene.block_center_y = 4.1f;
  scene.layer_gap = 0.02f;
  scene.buoyancy_strength = 1.25f;
  scene.phase_separation_strength = 2.5f;
  scene.bottom_material = MaterialType::Rock;
  scene.top_material = MaterialType::Water;

  sim.toggles.enable_gravity = true;
  sim.toggles.enable_stress = true;
  sim.toggles.enable_viscosity = true;
  sim.toggles.enable_buoyancy_helper = true;
  sim.toggles.enable_phase_separation_helper = true;
  sim.toggles.model_water_tait = true;
  sim.toggles.bc_left = true;
  sim.toggles.bc_right = true;
  sim.toggles.bc_bottom = true;
  sim.toggles.bc_top = false;

  // Bottom layer: lighter fluid.
  bottom.model = ConstitutiveModel::WeaklyCompressibleFluid;
  bottom.density0 = 700.f;
  bottom.bulk_modulus = 120.f;
  bottom.gamma = 4.f;
  bottom.viscosity = 0.006f;
  bottom.computeDerived();

  // Top layer: heavier fluid.
  top.model = ConstitutiveModel::WeaklyCompressibleFluid;
  top.density0 = 2200.f;
  top.bulk_modulus = 120.f;
  top.gamma = 4.f;
  top.viscosity = 0.006f;
  top.computeDerived();
}

static void refreshAfterReinitialize(TwoLayerDropSimulation &sim) {
  polyscope::removeStructure("two_layer_drop_particles", false);
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

  ImGui::Text("material : %s", kMaterialNames[(int)p.material]);
  ImGui::Text("pos      : (%.3f, %.3f)", p.pos.x(), p.pos.y());
  ImGui::Text("vel      : (%.3f, %.3f)", p.vel.x(), p.vel.y());
  ImGui::Text("J=det(F) : %.4f", p.F.determinant());
  ImGui::Text("F        : [[%.2f %.2f][%.2f %.2f]]", p.F(0, 0), p.F(0, 1),
              p.F(1, 0), p.F(1, 1));
  ImGui::Separator();
  ImGui::Text("rho      : %.1f", mp.density0);
  ImGui::Text("model    : %s", kModelNames[(int)mp.model]);
  ImGui::Text("bulk     : %.3e", mp.bulk_modulus);
  ImGui::Text("gamma    : %.3f", mp.gamma);
  ImGui::Text("visc     : %.3f", mp.viscosity);
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
  ImGui::SetNextWindowSize(ImVec2(390, 820), ImGuiCond_FirstUseEver);
  ImGui::Begin("Two-layer density inversion MPM debug");

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
  if (ImGui::Button("Apply default two-layer preset")) {
    applyDefaultTwoLayerPreset(*g_sim);
    p.computeDerived();
    g_sim->initialize();
    refreshAfterReinitialize(*g_sim);
    g_running = false;
  }
  if (ImGui::Button("Apply two-layer density inversion preset")) {
    applyTwoLayerDensityInversionPreset(*g_sim);
    p.computeDerived();
    g_sim->initialize();
    refreshAfterReinitialize(*g_sim);
    g_running = false;
  }

  ImGui::Separator();
  ImGui::TextWrapped(
      "This scene isolates a simpler two-layer falling column. It is meant to "
      "make density-driven inversion easier to see than in the four-layer "
      "drop. The inversion preset uses a lighter bottom fluid and a heavier "
      "top fluid, both modeled as weakly compressible Newtonian fluids. Two "
      "optional helper forces bias the motion toward buoyant inversion and "
      "visual phase separation.");

  ImGui::Separator();
  ImGui::Checkbox("Enable gravity", &t.enable_gravity);
  ImGui::Checkbox("Enable stress", &t.enable_stress);
  ImGui::Checkbox("Enable viscosity", &t.enable_viscosity);
  ImGui::Checkbox("Buoyancy helper", &t.enable_buoyancy_helper);
  ImGui::Checkbox("Phase-separation helper", &t.enable_phase_separation_helper);
  ImGui::Checkbox("Fluid EOS / pressure", &t.model_water_tait);

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
    ImGui::SliderFloat("buoyancy strength", &scene.buoyancy_strength, 0.0f,
                       4.0f);
    ImGui::SliderFloat("phase separation", &scene.phase_separation_strength,
                       0.0f, 6.0f);
    scene.block_w = std::max(scene.block_w, 0.1f);
    scene.block_h = std::max(scene.block_h, 0.1f);
    scene.layer_gap = std::max(scene.layer_gap, 0.0f);
    scene.buoyancy_strength = std::max(scene.buoyancy_strength, 0.0f);
    scene.phase_separation_strength =
        std::max(scene.phase_separation_strength, 0.0f);

    int bottom = static_cast<int>(scene.bottom_material);
    int top = static_cast<int>(scene.top_material);
    ImGui::Combo("Bottom material", &bottom, kMaterialNames,
                 IM_ARRAYSIZE(kMaterialNames));
    ImGui::Combo("Top material", &top, kMaterialNames,
                 IM_ARRAYSIZE(kMaterialNames));
    scene.bottom_material = static_cast<MaterialType>(bottom);
    scene.top_material = static_cast<MaterialType>(top);
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
        "Density edits affect newly initialized particle masses. Reset after "
        "large density edits so the block is re-seeded consistently.");
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
        "With the inversion preset, the heavier upper fluid should tend to "
        "sink through the lighter lower fluid, while the lighter fluid rises "
        "around it. The buoyancy helper compares a particle's rest density to "
        "the local neighborhood-average rest density, and the phase-separation "
        "helper pushes the two phases in opposite directions across the "
        "smoothed "
        "interface. These helpers are meant to make the inversion visibly "
        "happen "
        "in this simple MPM test; they are not a fully physical multiphase "
        "model.");
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
  params.dt = 1e-5f;
  params.gravity = -9.8f;
  params.computeDerived();

  TwoLayerDropSimulation sim(params);
  applyTwoLayerDensityInversionPreset(sim);
  sim.initialize();
  g_sim = &sim;

  polyscope::init();
  registerBoxBoundary(0.f, params.domain_w, 0.f, params.domain_h);

  polyscope::options::programName = "Two-layer density inversion MPM debug";
  polyscope::view::upDir = polyscope::UpDir::YUp;
  polyscope::view::style = polyscope::NavigateStyle::Planar;
  polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::None;

  sim.registerPolyscope();
  polyscope::view::lookAt({5., 3., 20.}, {5., 3., 0.}, {0., 1., 0.});
  polyscope::state::userCallback = uiCallback;
  polyscope::show();
  return 0;
}
