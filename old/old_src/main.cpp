#include "imgui.h"
#include "polyscope/curve_network.h"
#include "polyscope/polyscope.h"
#include "simulation.h"
#include <iostream>
#include <string>

static Simulation *g_sim = nullptr;
static bool g_running = false;
static int g_spf = 5;

static void registerBoxBoundary(double xmin = -1.0, double xmax = 1.0,
                                double ymin = 0.0, double ymax = 2.0) {
  // 4 corners (z = 0 for 2D)
  Eigen::MatrixXd nodes(4, 3);
  nodes << xmin, ymin, 0, xmax, ymin, 0, xmax, ymax, 0, xmin, ymax, 0;

  // 4 edges (connect corners)
  Eigen::MatrixXi edges(4, 2);
  edges << 0, 1, 1, 2, 2, 3, 3, 0;

  // Register with Polyscope
  auto *box = polyscope::registerCurveNetwork("box boundary", nodes, edges);

  // Optional styling
  box->setRadius(0.002);          // thickness of lines
  box->setColor({0.0, 0.0, 0.0}); // black
}

// ─────────────────────────────────────────────────────────────────────────────
//  Preset helper: applies a named configuration to the toggles so the user
//  can jump directly to a phase-equivalent state with one click.
// ─────────────────────────────────────────────────────────────────────────────
static void applyPreset(SimToggles &t, const char *name) {
  // Start from everything-off and enable what the named phase needs
  t = SimToggles{}; // reset all to defaults first

  if (std::string(name) == "Phase 1") {
    // Phase 1: particles visible, no physics at all.
    // P2G/G2P/advect still run so the sim doesn't freeze, but all forces
    // and constitutive models are off → particles fall as a pressureless gas
    // (actually they just sit still because gravity is also off).
    t.enable_gravity = false;
    t.enable_stress = false;
    t.enable_viscosity = false;
    t.model_water_tait = false;
    t.model_soil_elastic = false;
    t.model_sand_plastic = false;
    t.model_rock_elastic = false;

  } else if (std::string(name) == "Phase 2") {
    // Phase 2: gravity on, no stress forces.
    // Particles fall under gravity via the grid, but there's no pressure
    // or elastic resistance → layers compress through each other.
    t.enable_gravity = true;
    t.enable_stress = false;
    t.enable_viscosity = false;
    t.model_water_tait = false;
    t.model_soil_elastic = false;
    t.model_sand_plastic = false;
    t.model_rock_elastic = false;

  } else if (std::string(name) == "Phase 3") {
    // Phase 3: gravity + Tait EOS pressure for all materials.
    // All materials act as weakly compressible fluids: no shear resistance.
    t.enable_gravity = true;
    t.enable_stress = true;
    t.enable_viscosity = true;
    t.model_water_tait = true;
    t.model_soil_elastic = false; // soil acts as fluid
    t.model_sand_plastic = false; // sand acts as fluid, no yield surface
    t.model_rock_elastic = false; // rock acts as fluid
    // Note: soil/rock without elastic toggle still uses stressFluid-like
    // behaviour via the Tait fallback in kirchhoffStress

  } else if (std::string(name) == "Phase 4") {
    // Phase 4: full simulation: all models active.
    t = SimToggles{}; // all defaults = all on
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Particle inspector helper
// ─────────────────────────────────────────────────────────────────────────────
static void showParticleInfo(int idx) {
  if (idx < 0 || idx >= (int)g_sim->particles().size())
    return;
  const auto &p = g_sim->particles()[idx];
  float J = p.F.determinant();
  const char *names[] = {"Water", "Soil", "Sand", "Rock"};
  const char *models[] = {"Fluid(Tait)", "FixedCorotated", "DruckerPrager",
                          "?"};
  const MaterialParams &mp = g_sim->materialParams(p.material);
  int midx = (int)mp.model < 3 ? (int)mp.model : 3;
  ImGui::Text("material : %s  (%s)", names[(int)p.material], models[midx]);
  ImGui::Text("pos      : (%.3f, %.3f)", p.pos.x(), p.pos.y());
  ImGui::Text("vel      : (%.3f, %.3f)", p.vel.x(), p.vel.y());
  ImGui::Text("J=det(F) : %.4f", J);
  ImGui::Text("F        : [[%.2f %.2f][%.2f %.2f]]", p.F(0, 0), p.F(0, 1),
              p.F(1, 0), p.F(1, 1));
  if (p.material != MaterialType::Water) {
    ImGui::Text("E        : %.3e", mp.youngs_modulus);
    ImGui::Text("nu       : %.3f", mp.poisson_ratio);
  }
  if (p.material == MaterialType::Water) {
    ImGui::Text("rho      : %.1f", mp.density0);
    ImGui::Text("bulk     : %.3e", mp.bulk_modulus);
    ImGui::Text("gamma    : %.3f", mp.gamma);
    ImGui::Text("visc     : %.3e", mp.viscosity);
  } else if (p.material == MaterialType::Sand) {
    ImGui::Text("rho      : %.1f", mp.density0);
    ImGui::Text("phi      : %.2f", mp.friction_angle);
    ImGui::Text("cohesion : %.3e", mp.cohesion);
  } else {
    ImGui::Text("rho      : %.1f", mp.density0);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  UI callback: runs every frame
// ─────────────────────────────────────────────────────────────────────────────
void uiCallback() {
  SimToggles &t = g_sim->toggles;

  // ── Main control window ──────────────────────────────────────────────────
  ImGui::SetNextWindowPos(ImVec2(320, 10), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(310, 720), ImGuiCond_FirstUseEver);
  ImGui::Begin("MPM Debug: Phase 4");

  // ── Stats ─────────────────────────────────────────────────────────────────
  ImGui::Text("Particles : %zu", g_sim->particles().size());
  ImGui::Text("Frame     : %d", g_sim->frameCount());
  ImGui::Text("dt        : %.2e", g_sim->params().dt);
  ImGui::Separator();

  // ── Playback ─────────────────────────────────────────────────────────────
  if (ImGui::Button(g_running ? " Pause " : " Play  "))
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
  ImGui::SliderInt("Steps/frame", &g_spf, 1, 20);
  if (ImGui::Button("Reset")) {
    g_running = false;
    g_sim->initialize();
    g_sim->updatePolyscope();
  }
  ImGui::Separator();

  // ── Phase presets ─────────────────────────────────────────────────────────
  // One-click shortcuts to each phase's configuration.
  // These set the toggles below: you can then fine-tune from there.
  ImGui::TextDisabled("— Quick presets:");
  ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.20f, 0.35f, 0.55f, 1.f));
  if (ImGui::Button("  Phase 1  ")) {
    applyPreset(t, "Phase 1");
    g_sim->initialize();
    g_sim->updatePolyscope();
    g_running = false;
  }
  ImGui::PopStyleColor();
  ImGui::SameLine();
  ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.20f, 0.48f, 0.35f, 1.f));
  if (ImGui::Button("  Phase 2  ")) {
    applyPreset(t, "Phase 2");
    g_sim->initialize();
    g_sim->updatePolyscope();
    g_running = false;
  }
  ImGui::PopStyleColor();
  ImGui::SameLine();
  ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.55f, 0.38f, 0.10f, 1.f));
  if (ImGui::Button("  Phase 3  ")) {
    applyPreset(t, "Phase 3");
    g_sim->initialize();
    g_sim->updatePolyscope();
    g_running = false;
  }
  ImGui::PopStyleColor();
  ImGui::SameLine();
  ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.50f, 0.15f, 0.15f, 1.f));
  if (ImGui::Button("  Phase 4  ")) {
    applyPreset(t, "Phase 4");
    g_sim->initialize();
    g_sim->updatePolyscope();
    g_running = false;
  }
  ImGui::PopStyleColor();

  ImGui::Separator();

  // ── Phase 3 force toggles ─────────────────────────────────────────────────
  if (ImGui::CollapsingHeader("Phase 3: Forces",
                              ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::Indent(8.f);

    ImGui::Checkbox("Gravity", &t.enable_gravity);
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("Adds m*g to each grid node's force.\n"
                        "Off = particles float in zero-g.");

    ImGui::Checkbox("Stress forces", &t.enable_stress);
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("Scatters -vol*tau*grad_w to grid nodes.\n"
                        "Off = no pressure or elastic forces.\n"
                        "Equivalent to Phase 2 behaviour.");

    ImGui::Checkbox("Viscosity (water)", &t.enable_viscosity);
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("Adds mu*(C+C^T) to water stress.\n"
                        "Off = inviscid fluid (more sloshing).");

    ImGui::Unindent(8.f);
  }

  // ── Phase 4 constitutive model toggles ────────────────────────────────────
  if (ImGui::CollapsingHeader("Phase 4: Constitutive models",
                              ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::Indent(8.f);

    ImGui::TextDisabled("Water (blue)");
    ImGui::Checkbox("Tait EOS pressure##water", &t.model_water_tait);
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("p = k*(J^{-gamma}-1)\n"
                        "Off = water is pressureless gas.");
    ImGui::Spacing();
    ImGui::TextDisabled("Soil (green)");
    ImGui::Checkbox("Fixed-corotated elastic##soil", &t.model_soil_elastic);
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("tau = 2*mu*(F-R)*F^T + lambda*(J-1)*J*I\n"
                        "Off = soil acts as pressureless fluid.");

    ImGui::Spacing();
    ImGui::TextDisabled("Sand (orange)");
    ImGui::Checkbox("Drucker-Prager plasticity##sand", &t.model_sand_plastic);
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("Projects F^E onto the yield cone each step.\n"
                        "Off = sand is elastic (no flow, no angle of repose).\n"
                        "Compare: sand should pile at ~35 deg when ON.");

    ImGui::Spacing();
    ImGui::TextDisabled("Rock (red)");
    ImGui::Checkbox("Fixed-corotated elastic##rock", &t.model_rock_elastic);
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("Same model as soil but much stiffer E.\n"
                        "Off = rock acts as pressureless fluid.");

    ImGui::Unindent(8.f);
  }

  // ── Runtime-editable material parameters ─────────────────────────────────
  if (ImGui::CollapsingHeader("Material parameters",
                              ImGuiTreeNodeFlags_DefaultOpen)) {
    SimParams &p = g_sim->paramsMutable();

    auto drawElasticControls = [](const char *prefix, MaterialParams &mp) {
      ImGui::PushID(prefix);
      ImGui::InputFloat("Density", &mp.density0);
      ImGui::InputFloat("Young's modulus", &mp.youngs_modulus, 0.f, 0.f,
                        "%.3e");
      ImGui::SliderFloat("Poisson ratio", &mp.poisson_ratio, 0.0f, 0.45f);
      mp.density0 = std::max(mp.density0, 1.f);
      mp.youngs_modulus = std::max(mp.youngs_modulus, 1.f);
      ImGui::Text("mu      = %.3e", mp.mu);
      ImGui::Text("lambda  = %.3e", mp.lambda_lame);
      ImGui::PopID();
    };

    if (ImGui::TreeNode("Water")) {
      MaterialParams &mp = p.water_params;
      ImGui::InputFloat("Density##water", &mp.density0);
      ImGui::InputFloat("Bulk modulus##water", &mp.bulk_modulus, 0.f, 0.f,
                        "%.3e");
      ImGui::InputFloat("Gamma##water", &mp.gamma);
      ImGui::InputFloat("Viscosity##water", &mp.viscosity, 0.f, 0.f, "%.3e");
      mp.density0 = std::max(mp.density0, 1.f);
      mp.bulk_modulus = std::max(mp.bulk_modulus, 1.f);
      mp.gamma = std::max(mp.gamma, 0.1f);
      mp.viscosity = std::max(mp.viscosity, 0.f);
      ImGui::TreePop();
    }

    if (ImGui::TreeNode("Soil")) {
      drawElasticControls("soil", p.soil_params);
      ImGui::TreePop();
    }

    if (ImGui::TreeNode("Sand")) {
      MaterialParams &mp = p.sand_params;
      drawElasticControls("sand", mp);
      ImGui::InputFloat("Friction angle", &mp.friction_angle);
      ImGui::InputFloat("Cohesion", &mp.cohesion, 0.f, 0.f, "%.3e");
      mp.friction_angle = std::clamp(mp.friction_angle, 0.f, 89.f);
      mp.cohesion = std::max(mp.cohesion, 0.f);
      ImGui::Text("alpha_dp = %.3e", mp.alpha_dp);
      ImGui::TreePop();
    }

    if (ImGui::TreeNode("Rock")) {
      drawElasticControls("rock", p.rock_params);
      ImGui::TreePop();
    }

    p.computeDerived();
    ImGui::TextWrapped(
        "These values are now read directly by the constitutive-law code. "
        "Elastic/plastic behavior changes immediately; after changing density, "
        "press Reset so particle masses are rebuilt from the new densities.");
  }

  // ── Boundary condition toggles ────────────────────────────────────────────
  if (ImGui::CollapsingHeader("Boundary conditions")) {
    ImGui::Indent(8.f);
    ImGui::Checkbox("Left wall", &t.bc_left);
    ImGui::SameLine(120);
    ImGui::Checkbox("Right wall", &t.bc_right);
    ImGui::Checkbox("Bottom wall", &t.bc_bottom);
    ImGui::SameLine(120);
    ImGui::Checkbox("Top wall", &t.bc_top);
    ImGui::TextWrapped("Sticky walls: zeroes the inward velocity\n"
                       "component. Disable to see particles exit.");
    ImGui::Unindent(8.f);
  }

  // ── Particle inspector ────────────────────────────────────────────────────
  if (ImGui::CollapsingHeader("Particle inspector")) {
    static int inspect_idx = 0;
    ImGui::InputInt("Index", &inspect_idx);
    // Quick jumps to first particle of each material type
    if (ImGui::Button("First water")) {
      for (int i = 0; i < (int)g_sim->particles().size(); ++i)
        if (g_sim->particles()[i].material == MaterialType::Water) {
          inspect_idx = i;
          break;
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("First sand")) {
      for (int i = 0; i < (int)g_sim->particles().size(); ++i)
        if (g_sim->particles()[i].material == MaterialType::Sand) {
          inspect_idx = i;
          break;
        }
    }
    if (ImGui::Button("First soil")) {
      for (int i = 0; i < (int)g_sim->particles().size(); ++i)
        if (g_sim->particles()[i].material == MaterialType::Soil) {
          inspect_idx = i;
          break;
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("First rock")) {
      for (int i = 0; i < (int)g_sim->particles().size(); ++i)
        if (g_sim->particles()[i].material == MaterialType::Rock) {
          inspect_idx = i;
          break;
        }
    }
    ImGui::Separator();
    showParticleInfo(inspect_idx);
  }

  // ── What to expect box ───────────────────────────────────────────────────
  if (ImGui::CollapsingHeader("Expected behaviour guide")) {
    ImGui::TextWrapped("Phase 1 preset: particles sit still (no gravity,\n"
                       "  no stress). Verifies rendering.\n\n"
                       "Phase 2 preset: gravity ON, stress OFF.\n"
                       "  All materials fall and pile at bottom.\n"
                       "  Layers interpenetrate - no pressure.\n\n"
                       "Phase 3 preset: adds Tait pressure for all.\n"
                       "  Layers resist compression but all flow\n"
                       "  like fluids (no shear resistance).\n\n"
                       "Phase 4 preset: full simulation.\n"
                       "  Rock/soil resist shear (stiff elastic).\n"
                       "  Sand flows to angle of repose (~35 deg).\n"
                       "  Water sloshes.\n\n"
                       "Toggle Drucker-Prager OFF on sand to see\n"
                       "  elastic vs plastic sand behaviour.");
  }

  ImGui::End();

  // ── Tick ─────────────────────────────────────────────────────────────────
  if (g_running) {
    for (int i = 0; i < g_spf; ++i)
      g_sim->step();
    g_sim->updatePolyscope();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
int main() {
  SimParams params;
  params.domain_w = 10.f;
  params.domain_h = 6.f;
  params.layer_pct = 0.20f;
  params.ppc = 4;
  params.grid_nx = 80;
  params.grid_ny = 48;
  params.dt = 5e-5f;
  params.gravity = -9.8f;
  params.computeDerived();

  Simulation sim(params);
  sim.initialize();
  g_sim = &sim;

  polyscope::init();
  registerBoxBoundary(0.f, params.domain_w, 0.f, params.domain_h);
  polyscope::options::programName = "MPM Debug: Phase 4";
  polyscope::view::upDir = polyscope::UpDir::YUp;
  polyscope::view::style = polyscope::NavigateStyle::Planar;
  polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::None;

  sim.registerPolyscope();
  polyscope::view::lookAt({5., 3., 20.}, {5., 3., 0.}, {0., 1., 0.});
  polyscope::state::userCallback = uiCallback;
  polyscope::show();
  return 0;
}