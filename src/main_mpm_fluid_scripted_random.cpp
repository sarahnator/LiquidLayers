#include "imgui.h"
#include "mpm_fluid.h"
#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"
#include "polyscope/polyscope.h"
#include <algorithm>
#include <array>
#include <random>
#include <string>
#include <vector>

namespace {

FluidSimulation *g_sim = nullptr;
bool g_running = true;
int g_steps_per_frame = 3;
const char *kCloud = "particles";
std::mt19937 g_rng(12345);

struct ScheduledForce {
  int start = 0;
  int end = 0;
  MouseForceMode mode = MouseForceMode::GRAB;
  Eigen::Vector2f p0 = Eigen::Vector2f::Zero();
  Eigen::Vector2f p1 = Eigen::Vector2f::Zero();
  float radius = 1.f;
  float strength = 10.f;
};

ScheduledForce g_force;
bool g_force_active = false;
int g_next_event_frame = 40;
struct EventLogEntry {
  std::string short_msg;
  std::string full_msg;
};

bool g_debug_event_log = false;
std::vector<EventLogEntry> g_event_log;

std::string ff(float x, int p = 2) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(p) << x;
  return oss.str();
}

std::string vec2s(const Eigen::Vector2f &v, int p = 2) {
  std::ostringstream oss;
  oss << "(" << std::fixed << std::setprecision(p) << v.x() << ", " << v.y()
      << ")";
  return oss.str();
}

const char *modeName(MouseForceMode mode) {
  switch (mode) {
  case MouseForceMode::GRAB:
    return "GRAB";
  case MouseForceMode::WHIRL_LEGACY:
    return "WHIRL_LEGACY";
  case MouseForceMode::WHIRL_POOL:
    return "WHIRL_POOL";
  }
  return "UNKNOWN";
}

std::string fluidSummary(const FluidParams &f) {
  std::ostringstream oss;
  oss << f.name << "{rho0=" << ff(f.density0, 1)
      << ", K=" << ff(f.bulk_modulus, 1) << ", gamma=" << ff(f.gamma, 1)
      << ", mu=" << ff(f.viscosity, 3) << ", color=(" << ff(f.color[0]) << ", "
      << ff(f.color[1]) << ", " << ff(f.color[2]) << ")}";
  return oss.str();
}

std::string blockSummary(const BlockSpec &b, const FluidSimulation &sim) {
  const auto &bf = sim.fluidParams(b.bottom_fluid);
  const auto &tf = sim.fluidParams(b.top_fluid);
  std::ostringstream oss;
  oss << "block[x=" << ff(b.x_min) << ".." << ff(b.x_max)
      << ", y=" << ff(b.y_min) << ".." << ff(b.y_max)
      << ", gap=" << ff(b.layer_gap, 3) << ", bottom=" << fluidSummary(bf)
      << ", top=" << fluidSummary(tf) << "]";
  return oss.str();
}

std::string forceSummary(const ScheduledForce &burst) {
  std::ostringstream oss;
  oss << "action[mode=" << modeName(burst.mode) << ", frames=" << burst.start
      << "->" << burst.end << ", p0=" << vec2s(burst.p0)
      << ", p1=" << vec2s(burst.p1) << ", radius=" << ff(burst.radius)
      << ", strength=" << ff(burst.strength) << "]";
  return oss.str();
}

void logEvent(const std::string &short_msg, const std::string &full_msg = "") {
  g_event_log.push_back({short_msg, full_msg.empty() ? short_msg : full_msg});
  if (g_event_log.size() > 20)
    g_event_log.erase(g_event_log.begin());
}

void registerDomainBox(const SimParams &sp) {
  Eigen::MatrixXd nodes(4, 3);
  nodes << 0, 0, 0, sp.domain_w, 0, 0, sp.domain_w, sp.domain_h, 0, 0,
      sp.domain_h, 0;
  Eigen::MatrixXi edges(4, 2);
  edges << 0, 1, 1, 2, 2, 3, 3, 0;
  auto *box = polyscope::registerCurveNetwork("domain", nodes, edges);
  box->setRadius(0.002);
  box->setColor({1, 1, 1});
}

void rebuildCloud(const FluidSimulation &sim) {
  const auto &ps = sim.particles();
  std::vector<std::array<double, 3>> pts(ps.size()), cols(ps.size());
  for (size_t i = 0; i < ps.size(); ++i) {
    pts[i] = {(double)ps[i].pos.x(), (double)ps[i].pos.y(), 0.0};
    if (ps[i].fluid < sim.numFluids()) {
      const auto &c = sim.fluidParams(ps[i].fluid).color;
      cols[i] = {(double)c[0], (double)c[1], (double)c[2]};
    } else {
      cols[i] = {1.0, 0.0, 1.0};
    }
  }
  polyscope::removeStructure(kCloud, false);
  auto *cloud = polyscope::registerPointCloud(kCloud, pts);
  cloud->setPointRadius(0.0035);
  cloud->setPointRenderMode(polyscope::PointRenderMode::Sphere);
  cloud->setMaterial("flat");
  cloud->addColorQuantity("color", cols)->setEnabled(true);
}

void updateCloud(const FluidSimulation &sim) {
  const auto &ps = sim.particles();
  std::vector<std::array<double, 3>> pts(ps.size()), cols(ps.size());
  for (size_t i = 0; i < ps.size(); ++i) {
    pts[i] = {(double)ps[i].pos.x(), (double)ps[i].pos.y(), 0.0};
    if (ps[i].fluid < sim.numFluids()) {
      const auto &c = sim.fluidParams(ps[i].fluid).color;
      cols[i] = {(double)c[0], (double)c[1], (double)c[2]};
    } else {
      cols[i] = {1.0, 0.0, 1.0};
    }
  }
  auto *cloud = polyscope::getPointCloud(kCloud);
  cloud->updatePointPositions(pts);
  cloud->addColorQuantity("color", cols)->setEnabled(true);
}

float rf(float a, float b) {
  return std::uniform_real_distribution<float>(a, b)(g_rng);
}
int ri(int a, int b) { return std::uniform_int_distribution<int>(a, b)(g_rng); }

int secondsToFrames(const FluidSimulation &sim, float seconds) {
  const float dt = std::max(1e-6f, sim.params().dt);
  return std::max(1, (int)std::lround(seconds / dt));
}

static bool finiteVec(const Eigen::Vector2f &v) {
  return std::isfinite(v.x()) && std::isfinite(v.y());
}

void resetRegistry(FluidSimulation &sim) {
  while (sim.numFluids() > 1)
    sim.removeFluid(1);
  while (sim.numFluids() > 0)
    sim.removeFluid(0);

  FluidParams a;
  a.name = "Cyan";
  a.density0 = 900.f;
  a.bulk_modulus = 140.f;
  a.gamma = 4.f;
  a.viscosity = 0.010f;
  a.color = {0.05f, 0.95f, 1.00f};
  sim.addFluid(a);

  FluidParams b;
  b.name = "Dark";
  b.density0 = 2200.f;
  b.bulk_modulus = 260.f;
  b.gamma = 4.f;
  b.viscosity = 0.030f;
  b.color = {0.22f, 0.24f, 0.28f};
  sim.addFluid(b);

  FluidParams c;
  c.name = "Magenta";
  c.density0 = 650.f;
  c.bulk_modulus = 90.f;
  c.gamma = 4.f;
  c.viscosity = 0.020f;
  c.color = {1.00f, 0.25f, 0.85f};
  sim.addFluid(c);

  FluidParams d;
  d.name = "Yellow";
  d.density0 = 1200.f;
  d.bulk_modulus = 180.f;
  d.gamma = 4.f;
  d.viscosity = 0.015f;
  d.color = {1.00f, 0.95f, 0.10f};
  sim.addFluid(d);

  sim.paramsMutable().dt =
      std::min(sim.params().dt, sim.params().estimateDt(sim.allFluids()));
}

BlockSpec randomBlock(const SimParams &sp, int nf) {
  float w = rf(1.2f, 3.0f);
  float h = rf(0.8f, 1.8f);
  float x0 = rf(0.2f, std::max(0.21f, sp.domain_w - w - 0.2f));
  float y0 = rf(0.45f * sp.domain_h,
                std::max(0.46f * sp.domain_h, sp.domain_h - h - 0.2f));
  BlockSpec b;
  b.x_min = x0;
  b.x_max = x0 + w;
  b.y_min = y0;
  b.y_max = y0 + h;
  b.layer_gap = rf(0.0f, 0.04f);
  b.bottom_fluid = (FluidID)ri(0, nf - 1);
  b.top_fluid = (FluidID)ri(0, nf - 1);
  return b;
}

Eigen::Vector2f clampToDomain(const Eigen::Vector2f &p, const SimParams &sp,
                              float margin = 0.15f) {
  Eigen::Vector2f q = p;
  q.x() = std::clamp(q.x(), margin, sp.domain_w - margin);
  q.y() = std::clamp(q.y(), margin, sp.domain_h - margin);
  return q;
}

Eigen::Vector2f sampleParticleBiasedPoint(const FluidSimulation &sim,
                                          float jitter_radius = 1.0f) {
  const auto &ps = sim.particles();
  const SimParams &sp = sim.params();

  std::vector<int> valid;
  valid.reserve(ps.size());

  const float margin = std::max(0.35f, jitter_radius + 0.35f);

  for (int k = 0; k < (int)ps.size(); ++k) {
    const auto &p = ps[k];
    if (!finiteVec(p.pos) || !finiteVec(p.vel))
      continue;
    if (p.pos.x() < margin || p.pos.x() > sp.domain_w - margin)
      continue;
    if (p.pos.y() < margin || p.pos.y() > sp.domain_h - margin)
      continue;
    valid.push_back(k);
  }

  if (valid.empty()) {
    return {rf(0.25f * sp.domain_w, 0.75f * sp.domain_w),
            rf(0.35f * sp.domain_h, 0.80f * sp.domain_h)};
  }

  const auto &anchor = ps[valid[ri(0, (int)valid.size() - 1)]];

  const float angle = rf(0.f, 2.f * 3.14159265f);
  const float r = jitter_radius * std::sqrt(rf(0.f, 1.f));
  Eigen::Vector2f offset(r * std::cos(angle), r * std::sin(angle));

  return clampToDomain(anchor.pos + offset, sp, margin);
}

void scheduleRandomForceBurst(int frame) {
  const SimParams &sp = g_sim->params();

  g_force.start = frame;
  g_force.end = frame + ri(90, 220);

  g_force.mode =
      (ri(0, 1) == 0) ? MouseForceMode::GRAB : MouseForceMode::WHIRL_POOL;

  const float local_scale =
      (g_force.mode == MouseForceMode::WHIRL_POOL) ? 0.7f : 0.9f;

  g_force.p0 = sampleParticleBiasedPoint(*g_sim, local_scale);
  g_force.p1 = sampleParticleBiasedPoint(*g_sim, local_scale);

  if (g_force.mode == MouseForceMode::WHIRL_POOL) {
    g_force.radius = rf(0.6f, 1.1f);
    g_force.strength = rf(2.f, 5.f);
  } else {
    g_force.radius = rf(0.5f, 1.0f);
    g_force.strength = rf(2.f, 6.f);
  }

  g_force.p0 = sampleParticleBiasedPoint(*g_sim, local_scale);
  g_force.p1 = sampleParticleBiasedPoint(*g_sim, local_scale);

  if (!finiteVec(g_force.p0) || !finiteVec(g_force.p1)) {
    g_force_active = false;
    g_sim->clearMouseForce();
    logEvent("Skipped force burst: sampled non-finite mouse path");
    return;
  }

  g_force_active = true;

  const std::string short_msg = (g_force.mode == MouseForceMode::WHIRL_POOL)
                                    ? "Scheduled whirlpool burst"
                                    : "Scheduled grab sweep";

  logEvent(short_msg, short_msg + ": " + forceSummary(g_force));
  ;
}

void maybeSpawnRandomEvent() {
  const int f = g_sim->frame();
  if (f < g_next_event_frame)
    return;

  const int choice = ri(0, 4);
  bool spawned_block = false;
  if (choice <= 1) {
    BlockSpec b = randomBlock(g_sim->params(), g_sim->numFluids());
    g_sim->addBlock(b);
    rebuildCloud(*g_sim);
    logEvent("Dropped randomized block",
             "Dropped randomized block: " + blockSummary(b, *g_sim));
    spawned_block = true;

  } else if (choice <= 3) {
    scheduleRandomForceBurst(f);
  }
  // else {
  //   // Rare full reset so the random demo does not saturate forever.
  //   BlockSpec b = randomBlock(g_sim->params(), g_sim->numFluids());
  //   g_sim->initialize(b);
  //   rebuildCloud(*g_sim);
  //   logEvent("Random reset + new block");
  // }

  const float delay_sec = spawned_block ? rf(1.0f, 2.0f) : rf(0.0f, 2.0f);
  g_next_event_frame = f + secondsToFrames(*g_sim, delay_sec);
}

void applyCurrentRandomForce() {
  const int f = g_sim->frame();
  if (!g_force_active || f < g_force.start || f > g_force.end) {
    g_force_active = false;
    g_sim->clearMouseForce();
    return;
  }

  float alpha = 0.f;
  if (g_force.end > g_force.start) {
    alpha = float(f - g_force.start) / float(g_force.end - g_force.start);
  }
  Eigen::Vector2f pos = (1.f - alpha) * g_force.p0 + alpha * g_force.p1;
  g_sim->setMouseForce(pos, g_force.radius, g_force.strength, g_force.mode,
                       true);
}

void uiCallback() {
  ImGui::SetNextWindowPos({20.f, 20.f}, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize({420.f, 420.f}, ImGuiCond_FirstUseEver);
  ImGui::Begin("Random Script Demo");

  ImGui::Text("This executable plays a randomized scripted routine.");
  ImGui::Text("Seed: 12345");
  ImGui::Text("Frame: %d", g_sim->frame());
  ImGui::Text("Particles: %zu", g_sim->particles().size());
  ImGui::Text("Fluids: %d", g_sim->numFluids());
  ImGui::Text("Next event frame: %d", g_next_event_frame);
  ImGui::Separator();

  if (ImGui::Button(g_running ? "Pause" : "Play"))
    g_running = !g_running;
  ImGui::SameLine();
  if (ImGui::Button("Force new event"))
    g_next_event_frame = g_sim->frame();

  ImGui::SameLine();
  ImGui::Checkbox("Verbose event log", &g_debug_event_log);
  ImGui::SliderInt("Steps / frame", &g_steps_per_frame, 1, 20);
  ImGui::Separator();

  ImGui::TextWrapped("Random events include dropping blocks, moving grab "
                     "sweeps, and whirlpool "
                     "bursts.");

  ImGui::Separator();
  ImGui::Text("Recent events:");
  for (int i = (int)g_event_log.size() - 1; i >= 0; --i) {
    const std::string &msg =
        g_debug_event_log ? g_event_log[i].full_msg : g_event_log[i].short_msg;
    ImGui::BulletText("%s", msg.c_str());
  }

  ImGui::End();

  if (g_running) {
    for (int i = 0; i < g_steps_per_frame; ++i) {
      maybeSpawnRandomEvent();
      applyCurrentRandomForce();
      g_sim->step();
    }
    updateCloud(*g_sim);
  }
}

} // namespace

int main() {
  polyscope::init();
  auto [win_w, win_h] = polyscope::view::getWindowSize();
  const float aspect = (win_h > 0) ? (float)win_w / (float)win_h : (16.f / 9.f);

  SimParams sp;
  sp.domain_h = 6.f;
  sp.domain_w = sp.domain_h * aspect;
  sp.ppc = 4;
  sp.grid_ny = 48;
  sp.grid_nx = std::max(16, (int)(sp.grid_ny * aspect + 0.5f));
  sp.gravity = -9.8f;
  sp.computeDerived();

  FluidSimulation sim(sp);
  resetRegistry(sim);
  BlockSpec initial = randomBlock(sim.params(), sim.numFluids());
  sim.initialize(initial);
  g_sim = &sim;
  logEvent("Initialized random scripted demo");

  registerDomainBox(sim.params());
  polyscope::options::programName = "MPM Scripted Random Demo";
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
