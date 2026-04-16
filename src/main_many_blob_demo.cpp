#include "imgui.h"
#include "polyscope/curve_network.h"
#include "polyscope/polyscope.h"
#include "simulation_many_blob_demo.h"
#include <algorithm>

static ManyBlobSimulation *g_sim = nullptr;
static bool g_running = false;
static int g_spf = 5;

static const char *kModelNames[] = {"NewtonianFluid (WCMPM)"};

static void applyGrantStylePreset(ManyBlobSimulation &sim) {
  auto &p = sim.paramsMutable();
  auto &scene = sim.sceneParamsMutable();
  auto &a = sim.materialParamsMutable(MaterialType::Water);
  auto &b = sim.materialParamsMutable(MaterialType::Rock);

  p.domain_w = 7.5f;
  p.domain_h = 9.0f;
  p.grid_nx = 96;
  p.grid_ny = 115;
  p.ppc = 5;
  p.dt = 1.5e-5f;
  p.gravity = -9.8f;
  p.computeDerived();

  scene.num_blobs_x = 6;
  scene.num_blobs_y = 5;
  scene.blob_radius = 0.36f;
  scene.blob_spacing_x = 0.82f;
  scene.blob_spacing_y = 0.73f;
  scene.blob_center_x = 1.7f;
  scene.blob_center_y = 7.4f;
  scene.buoyancy_strength = 1.3f;
  scene.phase_separation_strength = 2.8f;
  scene.surface_tension_strength = 0.95f;
  scene.initial_swirl_speed = 0.55f;
  scene.initial_downward_speed = -0.3f;

  sim.toggles.enable_gravity = true;
  sim.toggles.enable_stress = true;
  sim.toggles.enable_viscosity = true;
  sim.toggles.model_water_tait = true;
  sim.toggles.model_water_freset = false;
  sim.toggles.bc_left = true;
  sim.toggles.bc_right = true;
  sim.toggles.bc_bottom = true;
  sim.toggles.bc_top = false;
  sim.toggles.enable_buoyancy_helper = true;
  sim.toggles.enable_phase_separation_helper = true;
  sim.toggles.enable_surface_tension_helper = true;

  a.model = ConstitutiveModel::WeaklyCompressibleFluid;
  a.density0 = 850.f;
  a.bulk_modulus = 110.f;
  a.gamma = 4.0f;
  a.viscosity = 0.008f;
  a.computeDerived();

  b.model = ConstitutiveModel::WeaklyCompressibleFluid;
  b.density0 = 2050.f;
  b.bulk_modulus = 110.f;
  b.gamma = 4.0f;
  b.viscosity = 0.008f;
  b.computeDerived();
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

static void showMaterialEditor(const char *label, MaterialParams &mp) {
  if (!ImGui::TreeNode(label))
    return;
  ImGui::PushID(label);

  int model = 0;
  ImGui::Combo("Constitutive model", &model, kModelNames, IM_ARRAYSIZE(kModelNames));
  mp.model = ConstitutiveModel::WeaklyCompressibleFluid;

  ImGui::InputFloat("Density", &mp.density0);
  ImGui::InputFloat("Bulk modulus", &mp.bulk_modulus, 0.f, 0.f, "%.3e");
  ImGui::SliderFloat("Gamma", &mp.gamma, 1.0f, 10.0f);
  ImGui::SliderFloat("Viscosity", &mp.viscosity, 0.0f, 1.0f);
  mp.density0 = std::max(mp.density0, 1.f);
  mp.bulk_modulus = std::max(mp.bulk_modulus, 1.f);
  mp.computeDerived();

  ImGui::PopID();
  ImGui::TreePop();
}

void uiCallback() {
  auto &t = g_sim->toggles;
  auto &p = g_sim->paramsMutable();
  auto &scene = g_sim->sceneParamsMutable();

  ImGui::SetNextWindowPos(ImVec2(320, 10), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(400, 820), ImGuiCond_FirstUseEver);
  ImGui::Begin("Many blob fluid MPM demo");

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
    g_sim->updatePolyscope();
  }
  if (ImGui::Button("Apply Grant-style preset")) {
    applyGrantStylePreset(*g_sim);
    g_sim->initialize();
    g_sim->updatePolyscope();
    g_running = false;
  }

  ImGui::Separator();
  ImGui::TextWrapped(
      "This demo spawns many round fluid blobs of two densities and two colors, "
      "with softer fluid parameters, open-top motion, and stylized helper "
      "forces to encourage clustering, separation, and richer large-scale "
      "motion. It is still MPM-based, but it is tuned more toward the look of "
      "an interactive liquid toy.");

  if (ImGui::CollapsingHeader("Simulation parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::InputFloat("dt", &p.dt, 0.f, 0.f, "%.2e");
    ImGui::InputFloat("gravity", &p.gravity);
    p.dt = std::max(p.dt, 1e-7f);
  }

  if (ImGui::CollapsingHeader("Blob layout", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::SliderInt("num blobs x", &scene.num_blobs_x, 1, 10);
    ImGui::SliderInt("num blobs y", &scene.num_blobs_y, 1, 10);
    ImGui::SliderFloat("blob radius", &scene.blob_radius, 0.1f, 0.7f);
    ImGui::SliderFloat("spacing x", &scene.blob_spacing_x, 0.2f, 1.5f);
    ImGui::SliderFloat("spacing y", &scene.blob_spacing_y, 0.2f, 1.5f);
    ImGui::InputFloat("center x", &scene.blob_center_x);
    ImGui::InputFloat("center y", &scene.blob_center_y);
    ImGui::SliderFloat("swirl", &scene.initial_swirl_speed, 0.0f, 2.0f);
    ImGui::SliderFloat("downward speed", &scene.initial_downward_speed, -2.0f, 0.0f);
  }

  if (ImGui::CollapsingHeader("Material parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
    showMaterialEditor("Light fluid (Water color)",
                       g_sim->materialParamsMutable(MaterialType::Water));
    showMaterialEditor("Heavy fluid (Rock color)",
                       g_sim->materialParamsMutable(MaterialType::Rock));
  }

  if (ImGui::CollapsingHeader("Helper forces", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::Checkbox("Buoyancy helper", &t.enable_buoyancy_helper);
    ImGui::Checkbox("Phase-separation helper", &t.enable_phase_separation_helper);
    ImGui::Checkbox("Surface-tension / cohesion helper", &t.enable_surface_tension_helper);
    ImGui::SliderFloat("buoyancy strength", &scene.buoyancy_strength, 0.0f, 4.0f);
    ImGui::SliderFloat("phase separation", &scene.phase_separation_strength, 0.0f, 6.0f);
    ImGui::SliderFloat("surface tension", &scene.surface_tension_strength, 0.0f, 3.0f);
  }

  if (ImGui::CollapsingHeader("Boundary conditions")) {
    ImGui::Checkbox("Left wall", &t.bc_left);
    ImGui::SameLine(150.f);
    ImGui::Checkbox("Right wall", &t.bc_right);
    ImGui::Checkbox("Bottom wall", &t.bc_bottom);
    ImGui::SameLine(150.f);
    ImGui::Checkbox("Top wall", &t.bc_top);
  }

  if (ImGui::CollapsingHeader("Expected behavior guide")) {
    ImGui::TextWrapped(
        "Compared to a single falling column, many separate blobs preserve more "
        "fluid area and create much richer collisions, merging, and interface "
        "motion. The two colors are both fluids with different rest densities. "
        "The helper forces are intentionally appearance-oriented: they are there "
        "to make separation and cohesion more visible in this MPM setting.");
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
  params.domain_w = 7.5f;
  params.domain_h = 9.0f;
  params.ppc = 5;
  params.grid_nx = 96;
  params.grid_ny = 115;
  params.dt = 1.5e-5f;
  params.gravity = -9.8f;
  params.computeDerived();

  ManyBlobSimulation sim(params);
  applyGrantStylePreset(sim);
  sim.initialize();
  g_sim = &sim;

  polyscope::init();
  registerBoxBoundary(0.f, params.domain_w, 0.f, params.domain_h);

  polyscope::options::programName = "Many blob fluid MPM demo";
  polyscope::view::upDir = polyscope::UpDir::YUp;
  polyscope::view::style = polyscope::NavigateStyle::Planar;
  polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::None;

  sim.registerPolyscope();
  polyscope::view::lookAt({3.75, 4.5, 20.}, {3.75, 4.5, 0.}, {0., 1., 0.});
  polyscope::state::userCallback = uiCallback;
  polyscope::show();
  return 0;
}
