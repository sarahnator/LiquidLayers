// =============================================================================
//  main_mpm_fluid.cpp
//
//  Polyscope / ImGui front end for the 2D MPM liquid demo.
//
//  Main UI features in the current code:
//    • runtime fluid registry editing
//    • block seeding with explicit top_fluid / bottom_fluid assignment
//    • mouse-driven grab / whirlpool interaction
//    • simple playback controls and particle inspection
//
//  Important note:
//    If you want a Rayleigh-Taylor-unstable drop,
//    assign the denser material to top_fluid directly.
// =============================================================================

#include "imgui.h"
#include "mpm_fluid.h"
#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"
#include "polyscope/polyscope.h"
#include <algorithm>
#include <string>
#include <tuple>
#include <vector>

static FluidSimulation *g_sim = nullptr;
static bool g_running = false;
static int g_spf = 5;
static const char *kCloud = "particles";

// Mouse-force UI state
static bool g_mouse_active = true;
static float g_mouse_radius = 0.9f;
static float g_mouse_strength = 12.f;
static int g_mouse_mode_idx = 0; // 0=legacy grab, 1=whirlpool
static Eigen::Vector2f g_mouse_domain_pos = Eigen::Vector2f::Zero();

// ─────────────────────────────────────────────────────────────────────────────
//  Preset builders
// ─────────────────────────────────────────────────────────────────────────────

// Returns a BlockSpec configured for a two-layer block.
// In the RT preset below, Fluid 0 is light and Fluid 1 is heavy, so assigning
// bottom_fluid=0 and top_fluid=1 makes the heavy phase start on top.
static BlockSpec makeRTBlock(const SimParams &p) {
  BlockSpec b;
  b.x_min = p.domain_w * 0.5f - 2.25f;
  b.x_max = p.domain_w * 0.5f + 2.25f;
  b.y_min = p.domain_h * 0.5f - 1.6f;
  b.y_max = p.domain_h * 0.5f + 1.6f;
  b.layer_gap = 0.02f;
  b.bottom_fluid = 0; // light phase in the lower band of the block
  b.top_fluid = 1;    // heavy phase in the upper band of the block
  return b;
}

static BlockSpec makeStableBlock(const SimParams &p) {
  BlockSpec b;
  b.x_min = p.domain_w * 0.5f - 2.25f;
  b.x_max = p.domain_w * 0.5f + 2.25f;
  b.y_min = p.domain_h * 0.5f - 1.6f;
  b.y_max = p.domain_h * 0.5f + 1.6f;
  b.layer_gap = 0.02f;
  b.bottom_fluid = 1; // heavy on bottom: stable
  b.top_fluid = 0;    // light on top
  return b;
}

static void applyRTPreset(FluidSimulation &sim) {
  // Rebuild the registry with a light and heavy phase.
  while (sim.numFluids() > 0)
    sim.removeFluid(0);

  FluidParams light;
  light.name = "Light";
  light.density0 = 700.f;
  light.bulk_modulus = 120.f;
  light.gamma = 4.f;
  light.viscosity = 0.006f;
  light.color = {0.05f, 0.95f, 1.00f};
  sim.addFluid(light); // FluidID 0

  FluidParams heavy;
  heavy.name = "Heavy";
  heavy.density0 = 2200.f;
  heavy.bulk_modulus = 260.f;
  heavy.gamma = 4.f;
  heavy.viscosity = 0.030f;
  heavy.color = {65.f / 255.f, 63.f / 255.f, 65.f / 255.f};
  sim.addFluid(heavy); // FluidID 1

  // Set dt from CFL using the new fluid speeds.
  SimParams &sp = sim.paramsMutable();
  //   sp.dt = sp.estimateDt(sim.allFluids());
}

static void applyStablePreset(FluidSimulation &sim) {
  while (sim.numFluids() > 0)
    sim.removeFluid(0);

  FluidParams light;
  light.name = "Light";
  light.density0 = 1000.f;
  light.bulk_modulus = 140.f;
  light.gamma = 4.f;
  light.viscosity = 0.010f;
  light.color = {0.05f, 0.95f, 1.00f};
  sim.addFluid(light);

  FluidParams heavy;
  heavy.name = "Heavy";
  heavy.density0 = 2200.f;
  heavy.bulk_modulus = 260.f;
  heavy.gamma = 4.f;
  heavy.viscosity = 0.030f;
  heavy.color = {65.f / 255.f, 63.f / 255.f, 65.f / 255.f};
  sim.addFluid(heavy);

  SimParams &sp = sim.paramsMutable();
  sp.dt = sp.estimateDt(sim.allFluids());
}

// ─────────────────────────────────────────────────────────────────────────────
//  Polyscope helpers
// ─────────────────────────────────────────────────────────────────────────────
static void registerDomainBox(const SimParams &sp) {
  Eigen::MatrixXd nodes(4, 3);
  nodes << 0, 0, 0, sp.domain_w, 0, 0, sp.domain_w, sp.domain_h, 0, 0,
      sp.domain_h, 0;
  Eigen::MatrixXi edges(4, 2);
  edges << 0, 1, 1, 2, 2, 3, 3, 0;
  auto *box = polyscope::registerCurveNetwork("domain", nodes, edges);
  box->setRadius(0.002);
  box->setColor({1, 1, 1});
}

static void rebuildCloud(const FluidSimulation &sim) {
  const auto &ps = sim.particles();
  std::vector<std::array<double, 3>> pts(ps.size()), cols(ps.size());
  for (size_t i = 0; i < ps.size(); ++i) {
    pts[i] = {(double)ps[i].pos.x(), (double)ps[i].pos.y(), 0.0};
    // Colors are looked up through the current registry. A magenta fallback is
    // shown if a particle ever carries an out-of-range FluidID.
    if (ps[i].fluid < sim.numFluids()) {
      const auto &c = sim.fluidParams(ps[i].fluid).color;
      cols[i] = {(double)c[0], (double)c[1], (double)c[2]};
    } else {
      cols[i] = {1, 0, 1}; // magenta = stale
    }
  }
  polyscope::removeStructure(kCloud, false);
  auto *cloud = polyscope::registerPointCloud(kCloud, pts);
  cloud->setPointRadius(0.0035);
  cloud->setPointRenderMode(polyscope::PointRenderMode::Sphere);
  cloud->setMaterial("flat");
  cloud->addColorQuantity("color", cols)->setEnabled(true);
}

static void updateCloud(const FluidSimulation &sim) {
  const auto &ps = sim.particles();
  std::vector<std::array<double, 3>> pts(ps.size()), cols(ps.size());
  for (size_t i = 0; i < ps.size(); ++i) {
    pts[i] = {(double)ps[i].pos.x(), (double)ps[i].pos.y(), 0.0};
    if (ps[i].fluid < sim.numFluids()) {
      const auto &c = sim.fluidParams(ps[i].fluid).color;
      cols[i] = {(double)c[0], (double)c[1], (double)c[2]};
    } else {
      cols[i] = {1, 0, 1};
    }
  }
  auto *cloud = polyscope::getPointCloud(kCloud);
  cloud->updatePointPositions(pts);
  cloud->addColorQuantity("color", cols)->setEnabled(true);
}

// ─────────────────────────────────────────────────────────────────────────────
//  UI state for the "drop block" panel
//
//  g_pending_block stores the next block the user intends to seed. The block is
//  only applied when the user presses one of the drop / replace buttons.
// ─────────────────────────────────────────────────────────────────────────────
static BlockSpec g_pending_block;
static bool g_pending_inited = false;

static void initPendingBlock(const SimParams &sp) {
  g_pending_block = makeRTBlock(sp);
  g_pending_inited = true;
}

static void prepareUnicodeImGuiFonts() {
  polyscope::options::prepareImGuiFontsCallback = []() {
    auto *atlas = new ImFontAtlas();

    static const ImWchar ranges[] = {0x0020, 0x00FF, // Basic Latin + Latin-1
                                     0x0370, 0x03FF, // Greek/Coptic
                                     0x2070, 0x209F, // Superscripts/Subscripts
                                     0};

    ImFontConfig cfg;
    auto addFirstAvailableFont = [&](const std::vector<const char *> &paths,
                                     float size) -> ImFont * {
      for (const char *path : paths) {
        if (ImFont *f = atlas->AddFontFromFileTTF(path, size, &cfg, ranges))
          return f;
      }
      return nullptr;
    };

    ImFont *regular = addFirstAvailableFont(
        {"/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
         "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
         "/System/Library/Fonts/Supplemental/Arial.ttf"},
        16.0f);

    ImFont *mono = addFirstAvailableFont(
        {"/System/Library/Fonts/Supplemental/Courier New.ttf",
         "/System/Library/Fonts/Supplemental/Menlo.ttc"},
        14.0f);

    if (!regular)
      regular = atlas->AddFontDefault();
    if (!mono)
      mono = regular;

    return std::make_tuple(atlas, regular, mono);
  };
}

// ─────────────────────────────────────────────────────────────────────────────
//  Mouse → domain coordinates
//
//  Polyscope's planar camera looks down -Z at the XY plane.
//  We unproject screen-space mouse from ImGui into world coords.
// ─────────────────────────────────────────────────────────────────────────────
static Eigen::Vector2f mouseToDomain() {
  ImVec2 mp = ImGui::GetMousePos();
  ImVec2 display_size = ImGui::GetIO().DisplaySize;
  float sx = mp.x / display_size.x;
  float sy = 1.f - mp.y / display_size.y;

  // Unproject via polyscope camera
  glm::vec4 viewport = {0.f, 0.f, (float)display_size.x, (float)display_size.y};
  glm::mat4 view = polyscope::view::getCameraViewMatrix();
  glm::mat4 proj = polyscope::view::getCameraPerspectiveMatrix();

  float px = mp.x, py = display_size.y - mp.y;
  glm::vec3 win_near(px, py, 0.f);
  glm::vec3 win_far(px, py, 1.f);
  glm::vec3 wn = glm::unProject(win_near, view, proj, viewport);
  glm::vec3 wf = glm::unProject(win_far, view, proj, viewport);

  // Intersect ray with z=0 plane
  float t = -wn.z / (wf.z - wn.z + 1e-12f);
  return Eigen::Vector2f(wn.x + t * (wf.x - wn.x), wn.y + t * (wf.y - wn.y));
}

// ─────────────────────────────────────────────────────────────────────────────
//  showFluidEditor() — inline material panel for one fluid slot
// ─────────────────────────────────────────────────────────────────────────────
static void showFluidEditor(FluidParams &fp, int id) {
  // Label uses the stored name
  char lbl[64];
  snprintf(lbl, sizeof(lbl), "Fluid %d: %s", id, fp.name.c_str());
  if (!ImGui::TreeNode(lbl))
    return;
  ImGui::PushID(id);

  static char name_buf[64];
  strncpy(name_buf, fp.name.c_str(), sizeof(name_buf) - 1);
  if (ImGui::InputText("Name", name_buf, sizeof(name_buf)))
    fp.name = name_buf;

  ImGui::ColorEdit3("Color", fp.color.data());
  ImGui::InputFloat("Density (ρ₀)", &fp.density0, 0, 0, "%.1f");
  ImGui::InputFloat("Bulk mod (K)", &fp.bulk_modulus, 0, 0, "%.3e");
  ImGui::SliderFloat("Gamma (γ)", &fp.gamma, 1.f, 10.f);
  ImGui::SliderFloat("Viscosity (μ)", &fp.viscosity, 0.f, 1.f);

  fp.density0 = std::max(fp.density0, 1.f);
  fp.bulk_modulus = std::max(fp.bulk_modulus, 1e-3f);
  fp.gamma = std::clamp(fp.gamma, 1.f, 12.f);
  fp.viscosity = std::max(fp.viscosity, 0.f);
  fp.computeDerived();
  ImGui::Text("Wave speed c₀ = %.2f m/s", fp.c0);

  ImGui::PopID();
  ImGui::TreePop();
}

// ─────────────────────────────────────────────────────────────────────────────
//  uiCallback()
// ─────────────────────────────────────────────────────────────────────────────
void uiCallback() {
  ImGui::SetNextWindowPos({320.f, 10.f}, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize({450.f, 900.f}, ImGuiCond_FirstUseEver);
  ImGui::Begin("Two-Fluid MPM");

  // ── Mouse-force update ───────────────────────────────────────────────────
  // Mouse forcing is only armed while the cursor is not over the GUI, and the
  // force is only actually applied while LMB is held down.
  bool hover_imgui = ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow) ||
                     ImGui::IsAnyItemActive();
  if (g_mouse_active && !hover_imgui) {
    g_mouse_domain_pos = mouseToDomain();
    MouseForceMode mode = MouseForceMode::GRAB;
    switch (g_mouse_mode_idx) {
    case 0:
      mode = MouseForceMode::GRAB;
      break;
    case 1:
      mode = MouseForceMode::WHIRL_POOL;
      break;
    default:
      mode = MouseForceMode::GRAB;
      break;
    }
    bool lmb_down = ImGui::IsMouseDown(0);
    g_sim->setMouseForce(g_mouse_domain_pos, g_mouse_radius, g_mouse_strength,
                         mode, lmb_down);
  } else {
    g_sim->clearMouseForce();
  }

  // ── Status ────────────────────────────────────────────────────────────────
  ImGui::Text("Particles : %zu", g_sim->particles().size());
  ImGui::Text("Frame     : %d", g_sim->frame());
  ImGui::Text("dt        : %.2e s", g_sim->params().dt);
  ImGui::Text("Fluids    : %d", g_sim->numFluids());
  ImGui::Separator();

  // ── Playback ──────────────────────────────────────────────────────────────
  if (ImGui::Button(g_running ? "Pause" : "Play"))
    g_running = !g_running;
  ImGui::SameLine();
  if (ImGui::Button("Step×1")) {
    g_sim->step();
    updateCloud(*g_sim);
  }
  ImGui::SameLine();
  if (ImGui::Button("Step×20")) {
    for (int i = 0; i < 20; ++i)
      g_sim->step();
    updateCloud(*g_sim);
  }
  ImGui::SliderInt("Steps/frame", &g_spf, 1, 30);
  ImGui::Separator();

  if (ImGui::Button("Reset (same fluids, same block)")) {
    g_running = false;
    g_sim->initialize(g_pending_block);
    rebuildCloud(*g_sim);
  }
  ImGui::Separator();

  // ── Simulation parameters ─────────────────────────────────────────────────
  if (ImGui::CollapsingHeader("Simulation parameters")) {
    SimParams &sp = g_sim->paramsMutable();
    ImGui::InputFloat("dt [s]", &sp.dt, 0, 0, "%.2e");
    ImGui::InputFloat("gravity", &sp.gravity);
    sp.dt = std::max(sp.dt, 1e-7f);

    float cfl = sp.estimateDt(g_sim->allFluids());
    ImGui::Text("CFL dt: %.2e s", cfl);
    // ImGui::SameLine();
    // if (ImGui::Button("Use CFL dt")) sp.dt = cfl;
  }

  // ── Mouse forces ──────────────────────────────────────────────────────────
  if (ImGui::CollapsingHeader("Mouse Forces", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::Checkbox("Enable mouse interaction", &g_mouse_active);

    if (g_mouse_active) {
      const char *modes[] = {"Grab (radial pull)",
                             "Whirlpool (spin + inward pull)"};
      ImGui::Combo("Mode", &g_mouse_mode_idx, modes, 2);

      if (g_mouse_mode_idx == 0) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.9f, 0.7f, 0.3f, 1.f));
        ImGui::TextWrapped(
            "Hold LMB to apply a radial pull toward the cursor.");
        ImGui::PopStyleColor();
      } else {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.45f, 0.85f, 1.0f, 1.f));
        ImGui::TextWrapped("Apply a vortex target field with "
                           "tangential spin plus inward "
                           "radial pull for a more whirlpool-like motion.");
        ImGui::PopStyleColor();
      }

      ImGui::SliderFloat("Radius [m]", &g_mouse_radius, 0.2f, 3.f);
      ImGui::SliderFloat("Strength", &g_mouse_strength, 1.f, 50.f);
      ImGui::Text("Cursor: (%.2f, %.2f)", g_mouse_domain_pos.x(),
                  g_mouse_domain_pos.y());
    }
  }

  // ── Fluid registry ────────────────────────────────────────────────────────
  if (ImGui::CollapsingHeader("Fluid registry",
                              ImGuiTreeNodeFlags_DefaultOpen)) {

    ImGui::TextWrapped(
        "Each registered fluid type has a FluidID (0-based index).  "
        "Particles keep their ID even if you edit parameters.  "
        "Removing a fluid remaps IDs above it downward.");

    for (int id = 0; id < g_sim->numFluids(); ++id) {
      showFluidEditor(g_sim->fluidParamsMutable((FluidID)id), id);
    }

    // ── Add fluid ─────────────────────────────────────────────────────────
    ImGui::Spacing();
    if (g_sim->numFluids() < kMaxFluids) {
      if (ImGui::Button("+ Add new fluid")) {
        FluidParams nf;
        nf.name = "New fluid " + std::to_string(g_sim->numFluids());

        auto neonFromHue = [](float hue) -> std::array<float, 3> {
          hue -= std::floor(hue);
          float h6 = hue * 6.f;
          int hi = (int)h6;
          float f = h6 - hi;
          float p = 0.10f;          // v=1, s=0.9
          float q = 1.f - 0.9f * f; // v*(1-s*f)
          float t = 1.f - 0.9f * (1 - f);
          switch (hi % 6) {
          case 0:
            return {1.f, t, p};
          case 1:
            return {q, 1.f, p};
          case 2:
            return {p, 1.f, t};
          case 3:
            return {p, q, 1.f};
          case 4:
            return {t, p, 1.f};
          default:
            return {1.f, p, q};
          }
        };

        auto colorDist2 = [](const std::array<float, 3> &a,
                             const std::array<float, 3> &b) {
          const float dr = a[0] - b[0];
          const float dg = a[1] - b[1];
          const float db = a[2] - b[2];
          return dr * dr + dg * dg + db * db;
        };

        // Pick the hue whose neon RGB is farthest from existing fluid colors.
        const int hue_samples = 96;
        float best_score = -1.f;
        std::array<float, 3> best_color = {1.f, 0.1f, 0.1f};
        for (int s = 0; s < hue_samples; ++s) {
          const float hue = (float)s / (float)hue_samples;
          const auto cand = neonFromHue(hue);

          float min_d2 = 1e9f;
          for (int fid = 0; fid < g_sim->numFluids(); ++fid) {
            const auto &c = g_sim->fluidParams((FluidID)fid).color;
            min_d2 = std::min(min_d2, colorDist2(cand, c));
          }

          if (min_d2 > best_score) {
            best_score = min_d2;
            best_color = cand;
          }
        }
        nf.color = best_color;
        int new_id = g_sim->addFluid(nf);
        if (new_id >= 0)
          std::cout << "[UI] Added fluid " << new_id << "\n";
      }
    } else {
      ImGui::TextDisabled("(registry full: %d/%d)", g_sim->numFluids(),
                          kMaxFluids);
    }

    // ── Remove fluid ──────────────────────────────────────────────────────
    if (g_sim->numFluids() > 1) {
      ImGui::SameLine();
      static int remove_id = 0;
      ImGui::SetNextItemWidth(60.f);
      ImGui::InputInt("##rmid", &remove_id, 0);
      remove_id = std::clamp(remove_id, 0, g_sim->numFluids() - 1);
      ImGui::SameLine();
      int count = g_sim->particleCountForFluid((FluidID)remove_id);
      char rm_lbl[64];
      snprintf(rm_lbl, sizeof(rm_lbl), "Remove fluid %d (%d particles)",
               remove_id, count);
      bool danger = count > 0;
      if (danger)
        ImGui::PushStyleColor(ImGuiCol_Button, {0.7f, 0.2f, 0.2f, 1.f});
      if (ImGui::Button(rm_lbl)) {
        g_sim->removeFluid((FluidID)remove_id);
        remove_id = std::clamp(remove_id, 0, g_sim->numFluids() - 1);
        rebuildCloud(*g_sim);
      }
      if (danger)
        ImGui::PopStyleColor();
      if (danger)
        ImGui::TextColored({1, 0.5f, 0.5f, 1},
                           "Warning: %d live particles use this fluid. "
                           "They will be remapped to ID %d after removal.",
                           count, remove_id);
    }
  }

  // ── Drop block panel ──────────────────────────────────────────────────────
  if (ImGui::CollapsingHeader("Drop a block", ImGuiTreeNodeFlags_DefaultOpen)) {
    if (!g_pending_inited)
      initPendingBlock(g_sim->params());
    BlockSpec &b = g_pending_block;

    ImGui::InputFloat("x min", &b.x_min);
    ImGui::SameLine();
    ImGui::InputFloat("x max", &b.x_max);
    ImGui::InputFloat("y min", &b.y_min);
    ImGui::SameLine();
    ImGui::InputFloat("y max", &b.y_max);
    ImGui::SliderFloat("layer gap", &b.layer_gap, 0.f, 0.1f);

    // Combo boxes for which fluid fills each half
    int nf = g_sim->numFluids();
    // Build the list of names used by the two fluid-selection combo boxes.
    std::vector<const char *> fnames;
    for (int i = 0; i < nf; ++i)
      fnames.push_back(g_sim->fluidParams((FluidID)i).name.c_str());
    fnames.push_back(nullptr); // sentinel

    int bot = (int)b.bottom_fluid, top = (int)b.top_fluid;
    ImGui::Combo("Top fluid", &top, fnames.data(), nf);
    ImGui::Combo("Bottom fluid", &bot, fnames.data(), nf);
    b.top_fluid = (FluidID)std::clamp(top, 0, nf - 1);
    b.bottom_fluid = (FluidID)std::clamp(bot, 0, nf - 1);

    // Clamp geometry to domain
    const SimParams &sp = g_sim->params();
    b.x_min = std::clamp(b.x_min, 0.001f, b.x_max - 0.01f);
    b.x_max = std::clamp(b.x_max, b.x_min + 0.01f, sp.domain_w - 0.001f);
    b.y_min = std::clamp(b.y_min, 0.001f, b.y_max - 0.01f);
    b.y_max = std::clamp(b.y_max, b.y_min + 0.01f, sp.domain_h - 0.001f);

    if (ImGui::Button("Drop block (adds to existing)")) {
      g_sim->addBlock(b);
      rebuildCloud(*g_sim);
    }
    ImGui::SameLine();
    if (ImGui::Button("Replace all with this block")) {
      g_sim->clearParticles();
      g_sim->addBlock(b);
      rebuildCloud(*g_sim);
    }
  }

  // ── Particle inspector ────────────────────────────────────────────────────
  if (ImGui::CollapsingHeader("Particle inspector")) {
    static int idx = 0;
    ImGui::InputInt("Index", &idx);
    idx = std::clamp(idx, 0, std::max(0, (int)g_sim->particles().size() - 1));
    if (!g_sim->particles().empty()) {
      const Particle &p = g_sim->particles()[idx];
      ImGui::Text("fluid  : %d (%s)", (int)p.fluid,
                  p.fluid < g_sim->numFluids()
                      ? g_sim->fluidParams(p.fluid).name.c_str()
                      : "STALE");
      ImGui::Text("pos    : (%.3f, %.3f)", p.pos.x(), p.pos.y());
      ImGui::Text("vel    : (%.3f, %.3f)", p.vel.x(), p.vel.y());
      ImGui::Text("J      : %.4f", p.J);
      ImGui::Text("mass   : %.4e", p.mass);
      ImGui::Text("vol0   : %.4e", p.vol0);
      if (p.fluid < g_sim->numFluids()) {
        const FluidParams &fp = g_sim->fluidParams(p.fluid);
        float J = std::clamp(p.J, kJ_min, kJ_max);
        float pr = fp.bulk_modulus * (std::exp(-fp.gamma * std::log(J)) - 1.f);
        pr = std::clamp(pr, -kPressureCap, kPressureCap);
        ImGui::Text("pressure (Tait): %.2f Pa", pr);
      }
    }
  }

  ImGui::End();

  if (g_running) {
    for (int i = 0; i < g_spf; ++i)
      g_sim->step();
    updateCloud(*g_sim);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────────────────────
int main() {
  prepareUnicodeImGuiFonts();
  polyscope::init();
  auto [win_w, win_h] = polyscope::view::getWindowSize();
  const float window_aspect =
      (win_h > 0) ? (float)win_w / (float)win_h : (16.f / 9.f);

  SimParams sp;
  sp.domain_h = 6.f;
  sp.domain_w = sp.domain_h * window_aspect;
  sp.ppc = 4;
  sp.grid_ny = 48;
  sp.grid_nx = std::max(16, (int)(sp.grid_ny * window_aspect + 0.5f));
  sp.gravity = -9.8f;
  sp.computeDerived();

  FluidSimulation sim(sp);
  applyRTPreset(sim);

  // You can optionally re-derive dt from the CFL estimate after the
  // fluids are registered. It is left manual here for easier experimentation.

  // Initialize with heavy fluid in the upper band, which is RT-unstable.
  BlockSpec block = makeRTBlock(sim.params());
  sim.initialize(block);
  g_sim = &sim;

  initPendingBlock(sim.params());

  registerDomainBox(sim.params());
  polyscope::options::programName = "MPM Fluid Demo";
  polyscope::view::upDir = polyscope::UpDir::YUp;
  polyscope::view::style = polyscope::NavigateStyle::Planar;
  polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::None;
  polyscope::options::buildDefaultGuiPanels = false;
  polyscope::options::openImGuiWindowForUserCallback = false;
  polyscope::view::bgColor = {0.f, 0.f, 0.f, 1.f};
  const double cx = 0.5 * sim.params().domain_w;
  const double cy = 0.5 * sim.params().domain_h;
  const double cz = 2.0 * std::max((double)sim.params().domain_w,
                                   (double)sim.params().domain_h);
  polyscope::view::lookAt({cx, cy, cz}, {cx, cy, 0.0}, {0., 1., 0.});

  rebuildCloud(sim);
  polyscope::state::userCallback = uiCallback;
  polyscope::show();
  return 0;
}
