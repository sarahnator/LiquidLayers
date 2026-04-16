#include "simulation.h"
#include "polyscope/point_cloud.h"
#include "polyscope/polyscope.h"
#include <algorithm>
#include <cmath>
#include <iostream>

Simulation::Simulation(SimParams params) : params_(std::move(params)) {
    params_.computeDerived();
    grid_.resize(params_.grid_nx * params_.grid_ny);
}

const MaterialParams& Simulation::materialParams(MaterialType m) const {
    switch (m) {
        case MaterialType::Water: return params_.water_params;
        case MaterialType::Soil:  return params_.soil_params;
        case MaterialType::Sand:  return params_.sand_params;
        case MaterialType::Rock:  return params_.rock_params;
    }
    return params_.water_params;
}
MaterialParams& Simulation::materialParamsMutable(MaterialType m) {
    switch (m) {
        case MaterialType::Water: return params_.water_params;
        case MaterialType::Soil:  return params_.soil_params;
        case MaterialType::Sand:  return params_.sand_params;
        case MaterialType::Rock:  return params_.rock_params;
    }
    return params_.water_params;
}

// ─────────────────────────────────────────────────────────────────────────────
//  initialize()
// ─────────────────────────────────────────────────────────────────────────────
void Simulation::initialize() {
    particles_.clear(); frame_ = 0;
    const float W=params_.domain_w, H=params_.domain_h,
                pct=params_.layer_pct, dx=params_.dx;
    const int ppc=params_.ppc;

    struct Layer { MaterialType mat; float y_lo, y_hi; };
    std::vector<Layer> layers = {
        {MaterialType::Rock,  0.f*H*pct, 1.f*H*pct},
        {MaterialType::Sand,  1.f*H*pct, 2.f*H*pct},
        {MaterialType::Soil,  2.f*H*pct, 3.f*H*pct},
        {MaterialType::Water, 3.f*H*pct, 4.f*H*pct},
    };

    const float px=dx/ppc, py=dx/ppc;
    auto jitter=[&](int i,float s)->float{
        return s*(static_cast<float>((i*1013904223+1664525)&0xFFFF)/65535.f-0.5f);};
    int pidx=0;
    for (const auto& l : layers) {
        const MaterialParams& mp=materialParams(l.mat);
        for (float y=l.y_lo+py*0.5f; y<l.y_hi; y+=py)
            for (float x=px*0.5f; x<W; x+=px) {
                Particle p;
                p.pos.x()=std::clamp(x+jitter(pidx*2,  px*0.3f),0.001f,W-0.001f);
                p.pos.y()=std::clamp(y+jitter(pidx*2+1,py*0.3f),0.001f,H-0.001f);
                p.material=l.mat;
                p.mass=mp.density0*px*py;
                p.vol0=px*py;
                particles_.push_back(p); ++pidx;
            }
    }
    free_surface_.assign(particles_.size(), false);
    std::cout<<"[MPM P5] "<<particles_.size()<<" particles\n";
}

// ─────────────────────────────────────────────────────────────────────────────
//  Constitutive models — identical to Phase 4 (your uploaded code)
// ─────────────────────────────────────────────────────────────────────────────
Eigen::Matrix2f Simulation::stressFluid(
    const Particle& p_const, const MaterialParams& mp) const
{
    Particle& p = const_cast<Particle&>(p_const);
    float J = p.F.determinant();
    J = std::clamp(J, 0.6f, 1.4f);
    if (toggles.model_water_freset)
        p.F = std::sqrt(J) * Eigen::Matrix2f::Identity();
    if (!toggles.model_water_tait)
        return Eigen::Matrix2f::Zero();
    float pressure = mp.bulk_modulus*(std::exp(-mp.gamma*std::log(J))-1.f);
    Eigen::Matrix2f tau = -J*pressure*Eigen::Matrix2f::Identity();
    if (toggles.enable_viscosity && mp.viscosity > 0.f)
        tau += J*mp.viscosity*(p.C + p.C.transpose());
    return tau;
}

Eigen::Matrix2f Simulation::stressFixedCorotated(
    const Particle& p, const MaterialParams& mp) const
{
    Eigen::Matrix2f R, S;
    polarDecompose2x2(p.F, R, S);
    float J = std::clamp(p.F.determinant(), 0.2f, 5.f);
    return 2.f*mp.mu*(p.F-R)*p.F.transpose()
           + mp.lambda_lame*(J-1.f)*J*Eigen::Matrix2f::Identity();
}

Eigen::Matrix2f Simulation::stressDruckerPrager(
    const Particle& p, const MaterialParams& mp) const
{
    return stressFixedCorotated(p, mp);
}

Eigen::Matrix2f Simulation::kirchhoffStress(const Particle& p) const {
    const MaterialParams& mp = materialParams(p.material);
    switch (mp.model) {
        case ConstitutiveModel::WeaklyCompressibleFluid:
            return stressFluid(p, mp);
        case ConstitutiveModel::FixedCorotated:
            if (p.material==MaterialType::Soil && !toggles.model_soil_elastic)
                return Eigen::Matrix2f::Zero();
            if (p.material==MaterialType::Rock  && !toggles.model_rock_elastic)
                return Eigen::Matrix2f::Zero();
            return stressFixedCorotated(p, mp);
        case ConstitutiveModel::DruckerPrager:
            return stressDruckerPrager(p, mp);
    }
    return Eigen::Matrix2f::Zero();
}

void Simulation::projectDruckerPrager(Particle& p,
                                       const MaterialParams& mp) const
{
    Eigen::Matrix2f U, V;
    Eigen::Vector2f sigma_vec;
    svd2x2(p.F, U, sigma_vec, V);
    sigma_vec(0)=std::max(sigma_vec(0),0.05f);
    sigma_vec(1)=std::max(sigma_vec(1),0.05f);
    Eigen::Vector2f eps(std::log(sigma_vec(0)),std::log(sigma_vec(1)));
    float tr_eps=eps(0)+eps(1);
    Eigen::Vector2f eps_dev=eps-(tr_eps/2.f)*Eigen::Vector2f::Ones();
    float dev_norm=eps_dev.norm();
    bool in_tension=(tr_eps>=0.f);
    float yield_value=dev_norm+mp.alpha_dp*tr_eps;
    if (!in_tension && yield_value<=0.f) return;
    Eigen::Vector2f eps_new;
    if (in_tension || dev_norm<1e-10f) {
        eps_new=Eigen::Vector2f::Zero();
    } else {
        float scale=-mp.alpha_dp*tr_eps/dev_norm;
        eps_new=scale*eps_dev+(tr_eps/2.f)*Eigen::Vector2f::Ones();
    }
    Eigen::Vector2f sigma_new(std::exp(eps_new(0)),std::exp(eps_new(1)));
    p.F=U*sigma_new.asDiagonal()*V.transpose();
}

// ─────────────────────────────────────────────────────────────────────────────
//  clearGrid()
// ─────────────────────────────────────────────────────────────────────────────
void Simulation::clearGrid() {
    for (auto& n : grid_) {
        n.mass=0.f;
        n.momentum=n.vel=n.vel_new=n.force=Eigen::Vector2f::Zero();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  substep_P2G()  — unchanged
// ─────────────────────────────────────────────────────────────────────────────
void Simulation::substep_P2G() {
    const float dx=params_.dx;
    const int nx=params_.grid_nx, ny=params_.grid_ny;
    for (const auto& p : particles_) {
        WeightStencil ws(p.pos,dx,nx,ny);
        for (int a=0;a<3;++a) for (int b=0;b<3;++b) {
            int ni=ws.base_i+a, nj=ws.base_j+b;
            if (!inGrid(ni,nj)) continue;
            float w=ws.weight(a,b);
            Eigen::Vector2f xip=Eigen::Vector2f(ni*dx,nj*dx)-p.pos;
            auto& node=grid_[gridIdx(ni,nj)];
            node.mass     += w*p.mass;
            node.momentum += w*p.mass*(p.vel+p.C*xip);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  substep_volumeRecompute()
//
//  Phase 5 addition.  Called immediately after P2G (when grid has fresh mass).
//
//  Problem it solves:
//    vol0 is set at initialization to px*py and never updated.  As particles
//    spread out near the free surface, the force formula  f = -vol0*tau*grad_w
//    underestimates the volume each particle actually represents, so free-surface
//    particles generate too little pressure to push their neighbours away and
//    cluster into visible "blobs" before vanishing.
//
//  Fix:
//    After P2G, the grid node at (i,j) holds:
//        node.mass  = sum_p  w_ip * m_p         (units: kg)
//    The density the grid "sees" at node (i,j) is:
//        rho_i      = node.mass / dx^2           (units: kg/m^2 in 2D)
//    We gather this density back to each particle using the same weights:
//        rho_p      = sum_i  w_ip * (node.mass / dx^2)
//    Then the particle's current volume is:
//        vol_p      = m_p / rho_p
//    This vol_p replaces vol0 for this step's stress force scatter.
//
//  Why do we use node.mass/dx^2 and not node.mass directly?
//    node.mass has units of kg.  To get density (kg/m^2) we divide by the
//    cell area dx^2.  This is the same conversion you'd do in SPH when
//    estimating density from neighbour masses.
//
//  Stability note:  we clamp rho_p away from zero to avoid division by zero
//    at the free surface where some grid nodes genuinely have zero mass.
//    The minimum density is set to 10% of the particle's own rest density.
// ─────────────────────────────────────────────────────────────────────────────
void Simulation::substep_volumeRecompute() {
    const float dx=params_.dx, dx2=dx*dx;
    const int nx=params_.grid_nx, ny=params_.grid_ny;

    for (auto& p : particles_) {
        const MaterialParams& mp=materialParams(p.material);

        // Gather density estimate from grid
        float rho_p = 0.f;
        WeightStencil ws(p.pos,dx,nx,ny);
        for (int a=0;a<3;++a) for (int b=0;b<3;++b) {
            int ni=ws.base_i+a, nj=ws.base_j+b;
            if (!inGrid(ni,nj)) continue;
            float w=ws.weight(a,b);
            // node.mass / dx^2 is the grid-estimated mass density at this node
            rho_p += w * (grid_[gridIdx(ni,nj)].mass / dx2);
        }

        // Clamp to at least 10% of rest density to avoid singularities
        float rho_min = 0.1f * mp.density0;
        rho_p = std::max(rho_p, rho_min);

        // Update volume estimate
        p.vol0 = p.mass / rho_p;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  substep_gridUpdate()
//
//  Phase 5 additions:
//    - Mouse force: a smooth radial body force added to grid nodes near cursor
//    - Thicker wall (wall=3) and position margin (3*dx) to stop vacuum instability
// ─────────────────────────────────────────────────────────────────────────────
void Simulation::substep_gridUpdate() {
    const float dx=params_.dx, dt=params_.dt;
    const int nx=params_.grid_nx, ny=params_.grid_ny;

    // Phase 5: thicker wall — 3 nodes instead of 2
    const int wall=3;

    // Pass 1: stress forces
    if (toggles.enable_stress) {
        for (const auto& p : particles_) {
            Eigen::Matrix2f tau=kirchhoffStress(p);
            WeightStencil ws(p.pos,dx,nx,ny);
            for (int a=0;a<3;++a) for (int b=0;b<3;++b) {
                int ni=ws.base_i+a, nj=ws.base_j+b;
                if (!inGrid(ni,nj)) continue;
                grid_[gridIdx(ni,nj)].force -=
                    p.vol0*(tau*ws.weightGrad(a,b,dx));
            }
        }
    }

    // Pass 2: velocity update + gravity + mouse force + BCs
    for (int j=0;j<ny;++j) for (int i=0;i<nx;++i) {
        auto& node=grid_[gridIdx(i,j)];
        if (node.mass<1e-10f) continue;

        node.vel=node.momentum/node.mass;

        if (toggles.enable_gravity)
            node.force.y()+=node.mass*params_.gravity;

        // ── Phase 5: mouse force ──────────────────────────────────────────────
        //
        //  Model:  f_i^mouse = w_mouse(x_i) * strength * (x_mouse - x_i)
        //
        //  w_mouse is a quadratic tent kernel:
        //    w(r) = max(0, 1 - r/R)^2    where r = ||x_i - x_mouse||
        //  This gives a smooth falloff from the cursor with:
        //    - Maximum force at the cursor itself (r=0)
        //    - Zero force at distance R (the interaction radius)
        //    - Continuous first derivative (no force discontinuity)
        //
        //  The direction (x_mouse - x_i) makes it attractive when strength>0
        //  (node is pulled toward cursor) and repulsive when strength<0.
        //
        //  We apply force to grid nodes rather than particles directly so that
        //  the interaction is consistent with how all other forces work in MPM:
        //  forces live on the grid and propagate to particles via G2P.
        //  Applying force directly to particles would bypass the grid entirely
        //  and break momentum conservation.
        // ─────────────────────────────────────────────────────────────────────
        if (toggles.enable_mouse_force && mouse.active) {
            float xi = i * dx;
            float yi = j * dx;
            float rx = xi - mouse.x;
            float ry = yi - mouse.y;
            float r  = std::sqrt(rx*rx + ry*ry);

            if (r < mouse.radius) {
                // Quadratic tent kernel weight
                float t = 1.f - r/mouse.radius;
                float w = t * t;

                // Force direction: from node toward cursor (attractive)
                // Magnitude scales with weight and user-controlled strength
                Eigen::Vector2f dir(mouse.x - xi, mouse.y - yi);

                // Normalise direction only if non-zero (avoid NaN at r=0)
                if (r > 1e-6f) dir /= r;

                node.force += node.mass * mouse.strength * w * dir;
            }
        }

        node.vel_new=node.vel+dt*node.force/node.mass;

        // BCs with thicker wall
        if (toggles.bc_left   && i<wall      && node.vel_new.x()<0.f) node.vel_new.x()=0.f;
        if (toggles.bc_right  && i>=nx-wall  && node.vel_new.x()>0.f) node.vel_new.x()=0.f;
        if (toggles.bc_bottom && j<wall      && node.vel_new.y()<0.f) node.vel_new.y()=0.f;
        if (toggles.bc_top    && j>=ny-wall  && node.vel_new.y()>0.f) node.vel_new.y()=0.f;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  substep_G2P()  — unchanged from Phase 4
// ─────────────────────────────────────────────────────────────────────────────
void Simulation::substep_G2P() {
    const float dx=params_.dx, D_inv=params_.D_inv, dt=params_.dt;
    const int nx=params_.grid_nx, ny=params_.grid_ny;

    for (auto& p : particles_) {
        WeightStencil ws(p.pos,dx,nx,ny);
        Eigen::Vector2f v_new=Eigen::Vector2f::Zero();
        Eigen::Matrix2f C_new=Eigen::Matrix2f::Zero();
        for (int a=0;a<3;++a) for (int b=0;b<3;++b) {
            int ni=ws.base_i+a, nj=ws.base_j+b;
            if (!inGrid(ni,nj)) continue;
            float w=ws.weight(a,b);
            const Eigen::Vector2f& vi=grid_[gridIdx(ni,nj)].vel_new;
            Eigen::Vector2f xip=Eigen::Vector2f(ni*dx,nj*dx)-p.pos;
            v_new+=w*vi; C_new+=w*(vi*xip.transpose());
        }
        p.vel=v_new; p.C=D_inv*C_new;
        p.F=(Eigen::Matrix2f::Identity()+dt*p.C)*p.F;

        const MaterialParams& mp=materialParams(p.material);
        if (mp.model==ConstitutiveModel::WeaklyCompressibleFluid) {
            if (toggles.model_water_freset) {
                float J=std::clamp(p.F.determinant(),0.6f,1.4f);
                p.F=std::sqrt(J)*Eigen::Matrix2f::Identity();
            }
        } else if (mp.model==ConstitutiveModel::DruckerPrager) {
            if (toggles.model_sand_plastic)
                projectDruckerPrager(p,mp);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  substep_advect()
//
//  Phase 5: clamp margin increased from 0.001 to wall*dx.
//  This keeps every particle's 3x3 stencil fully inside the grid,
//  preventing the vacuum instability at the domain boundary.
//
//  Why wall*dx?
//    base_i = floor(x/dx - 0.5)  (from WeightStencil constructor)
//    The leftmost stencil node is at  base_i * dx.
//    For base_i >= 0 we need  x/dx - 0.5 >= 0  =>  x >= 0.5*dx.
//    With wall=3 we also need the stencil to stay away from the wall zone,
//    so we use margin = wall*dx which gives comfortable clearance.
// ─────────────────────────────────────────────────────────────────────────────
void Simulation::substep_advect() {
    const float dt=params_.dt, W=params_.domain_w, H=params_.domain_h;
    const float margin = 3.f * params_.dx;   // Phase 5: wider margin
    for (auto& p : particles_) {
        p.pos+=dt*p.vel;
        p.pos.x()=std::clamp(p.pos.x(), margin, W-margin);
        p.pos.y()=std::clamp(p.pos.y(), margin, H-margin);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  substep_freeSurface()
//
//  Phase 5 addition.  Tags particles that are near the free surface.
//  Called after P2G so the grid has fresh mass data.
//
//  A particle p is tagged as a free-surface particle if the total grid mass
//  visible to it (weighted by its own B-spline weights) is less than a
//  fraction theta of its own mass:
//
//    sum_i  w_ip * node.mass  <  theta * m_p
//
//  Interpretation:
//    If the weighted sum of nearby node masses equals m_p, the particle is
//    fully "surrounded" by material — it's interior.  If the sum is much
//    less, the particle is at the edge of the material where grid cells are
//    only partially filled — it's on the free surface.
//
//  theta = 0.5 means: tag if less than half the expected neighbourhood mass
//  is present.  Lower theta = only tag very isolated particles.
//  Higher theta = tag more aggressively (includes thin interior regions too).
//
//  This information is used for:
//    1. Visualization: color free-surface particles differently
//    2. Debugging: seeing where the vacuum instability is forming
//    3. Future: could be used to apply surface tension forces only at surface
// ─────────────────────────────────────────────────────────────────────────────
void Simulation::substep_freeSurface() {
    const float dx=params_.dx;
    const int nx=params_.grid_nx, ny=params_.grid_ny;
    const float theta=0.5f;   // surface detection threshold

    free_surface_.resize(particles_.size());

    for (size_t pidx=0; pidx<particles_.size(); ++pidx) {
        const auto& p=particles_[pidx];
        WeightStencil ws(p.pos,dx,nx,ny);

        float neighbourhood_mass=0.f;
        for (int a=0;a<3;++a) for (int b=0;b<3;++b) {
            int ni=ws.base_i+a, nj=ws.base_j+b;
            if (!inGrid(ni,nj)) continue;
            neighbourhood_mass += ws.weight(a,b) * grid_[gridIdx(ni,nj)].mass;
        }

        free_surface_[pidx] = (neighbourhood_mass < theta * p.mass);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  step()
//
//  Phase 5 step order:
//    clearGrid -> P2G -> [volumeRecompute] -> [freeSurface] -> gridUpdate
//    -> G2P -> advect
//
//  volumeRecompute and freeSurface both need fresh P2G grid data, so they
//  run right after P2G and before gridUpdate modifies the grid.
// ─────────────────────────────────────────────────────────────────────────────
void Simulation::step() {
    clearGrid();
    substep_P2G();

    if (toggles.enable_vol_recompute)
        substep_volumeRecompute();   // P5: update vol0 from grid density

    if (toggles.show_free_surface)
        substep_freeSurface();        // P5: tag surface particles for coloring

    substep_gridUpdate();
    substep_G2P();
    substep_advect();
    ++frame_;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Polyscope
//
//  Phase 5: when show_free_surface is on, free-surface particles are rendered
//  white instead of their material color so they're immediately visible.
// ─────────────────────────────────────────────────────────────────────────────
void Simulation::buildRenderArrays(
    std::vector<std::array<double,3>>& pos3d,
    std::vector<std::array<double,3>>& colors) const
{
    pos3d.resize(particles_.size());
    colors.resize(particles_.size());
    for (size_t i=0;i<particles_.size();++i) {
        const auto& p=particles_[i];
        pos3d[i]={(double)p.pos.x(),(double)p.pos.y(),0.};

        // Free-surface highlighting: white for tagged particles
        if (toggles.show_free_surface
            && i < free_surface_.size()
            && free_surface_[i]) {
            colors[i]={1.,1.,1.};
        } else {
            auto c=materialColor(p.material);
            colors[i]={c[0],c[1],c[2]};
        }
    }
}
void Simulation::registerPolyscope() {
    std::vector<std::array<double,3>> pos3d,colors;
    buildRenderArrays(pos3d,colors);
    auto* cloud=polyscope::registerPointCloud(kCloudName,pos3d);
    cloud->setPointRadius(0.003);
    cloud->setPointRenderMode(polyscope::PointRenderMode::Sphere);
    cloud->addColorQuantity("material_color",colors)->setEnabled(true);
}
void Simulation::updatePolyscope() {
    std::vector<std::array<double,3>> pos3d,colors;
    buildRenderArrays(pos3d,colors);
    auto* cloud=polyscope::getPointCloud(kCloudName);
    cloud->updatePointPositions(pos3d);
    cloud->addColorQuantity("material_color",colors)->setEnabled(true);
}
