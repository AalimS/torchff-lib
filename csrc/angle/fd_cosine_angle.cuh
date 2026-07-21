#ifndef TORCHFF_FD_COSINE_ANGLE_CUH
#define TORCHFF_FD_COSINE_ANGLE_CUH

#include "common/vec3.cuh"

template <typename scalar_t>
__device__ __forceinline__ void fd_cos_angle(
    scalar_t rij_x, scalar_t rij_y, scalar_t rij_z,  // vec B->A
    scalar_t rkj_x, scalar_t rkj_y, scalar_t rkj_z,  // vec B->C
    scalar_t efx, scalar_t efy, scalar_t efz,          // E-field at B
    scalar_t th0,                                       // equilibrium angle
    scalar_t k0,                                        // force constant
    scalar_t dmu_dtheta,                                // |dmu/dtheta|
    scalar_t d2mu_dtheta2,                              // |d2mu/dtheta2|
    scalar_t* ene,
    // NEW: dc = cos_theta - cos_theta0_eff, needed by caller for bb/ba coupling
    scalar_t* dc_out,
    // NEW: dcos/dr exposed so caller can reuse for bb/ba forces without recomputing
    scalar_t* dcos_dri_x_out, scalar_t* dcos_dri_y_out, scalar_t* dcos_dri_z_out,
    scalar_t* dcos_drk_x_out, scalar_t* dcos_drk_y_out, scalar_t* dcos_drk_z_out,
    // NEW: d(dc)/d(eps) = -dcos_t0eff_deps, needed by caller for ba efield grad
    scalar_t* ddc_deps_out,
    // NEW: unit bisector d_hat, needed by caller to project ba efield grad
    scalar_t* d_hat_x_out, scalar_t* d_hat_y_out, scalar_t* d_hat_z_out,
    scalar_t* dri_x, scalar_t* dri_y, scalar_t* dri_z,  // grad w.r.t. atom i (A)
    scalar_t* drk_x, scalar_t* drk_y, scalar_t* drk_z,  // grad w.r.t. atom k (C)
    scalar_t* def_x, scalar_t* def_y, scalar_t* def_z   // grad w.r.t. E-field at B
)
{
    // --- Geometry ---
    scalar_t rij = sqrt_(rij_x*rij_x + rij_y*rij_y + rij_z*rij_z);
    scalar_t rkj = sqrt_(rkj_x*rkj_x + rkj_y*rkj_y + rkj_z*rkj_z);

    scalar_t rij_inv = scalar_t(1.0) / rij;
    scalar_t rkj_inv = scalar_t(1.0) / rkj;

    scalar_t uij_x = rij_x * rij_inv;
    scalar_t uij_y = rij_y * rij_inv;
    scalar_t uij_z = rij_z * rij_inv;

    scalar_t ukj_x = rkj_x * rkj_inv;
    scalar_t ukj_y = rkj_y * rkj_inv;
    scalar_t ukj_z = rkj_z * rkj_inv;

    scalar_t b_x = uij_x + ukj_x;
    scalar_t b_y = uij_y + ukj_y;
    scalar_t b_z = uij_z + ukj_z;
    scalar_t b_norm = sqrt_(b_x*b_x + b_y*b_y + b_z*b_z);
    scalar_t b_norm_inv = scalar_t(1.0) / b_norm;

    scalar_t d_x = b_x * b_norm_inv;
    scalar_t d_y = b_y * b_norm_inv;
    scalar_t d_z = b_z * b_norm_inv;

    // NEW: expose d_hat so caller can use it for ba efield gradient
    *d_hat_x_out = d_x;
    *d_hat_y_out = d_y;
    *d_hat_z_out = d_z;

    scalar_t eps = efx*d_x + efy*d_y + efz*d_z;

    scalar_t dot = rij_x*rkj_x + rij_y*rkj_y + rij_z*rkj_z;
    scalar_t cos_theta = dot * rij_inv * rkj_inv;
    constexpr scalar_t ONE  = scalar_t(0.9999999999);
    constexpr scalar_t MONE = scalar_t(-0.9999999999);
    cos_theta = clamp_(cos_theta, MONE, ONE);

    // --- FD parameters ---
    scalar_t sin_t0  = sin_(th0);
    scalar_t cos_t0  = cos_(th0);
    scalar_t sin2_t0 = sin_t0 * sin_t0;
    scalar_t cot_t0  = cos_t0 / sin_t0;

    scalar_t mu_prime  = dmu_dtheta   * eps;
    scalar_t mu_dprime = d2mu_dtheta2 * eps;

    scalar_t lambda1 = -mu_prime / sin_t0;
    scalar_t lambda2 = (mu_dprime - cot_t0 * mu_prime) / sin2_t0;

    scalar_t k_eff = k0 - lambda2;
    scalar_t k_eff_min = scalar_t(0.4) * k0;
    bool clamped = k_eff < k_eff_min;
    k_eff = clamped ? k_eff_min : k_eff;

    scalar_t cos_t0_eff = cos_t0 + lambda1 / k_eff;
    cos_t0_eff = clamp_(cos_t0_eff, scalar_t(-1.0 + 1e-6), scalar_t(1.0 - 1e-6));

    scalar_t dc = cos_theta - cos_t0_eff;

    // NEW: expose dc so caller can use it for ba coupling (V_ba = k_ba * dr * dc)
    *dc_out = dc;

    // --- Energy ---
    *ene = scalar_t(0.5) * k_eff * dc * dc;

    // --- dV/d(cos_theta): direct geometric term ---
    scalar_t dV_dcos = k_eff * dc;

    scalar_t tmp = rij * rkj;
    scalar_t dcos_dri_x = (rkj_x - dot*rij_inv*rij_inv*rij_x) / tmp;
    scalar_t dcos_dri_y = (rkj_y - dot*rij_inv*rij_inv*rij_y) / tmp;
    scalar_t dcos_dri_z = (rkj_z - dot*rij_inv*rij_inv*rij_z) / tmp;
    scalar_t dcos_drk_x = (rij_x - dot*rkj_inv*rkj_inv*rkj_x) / tmp;
    scalar_t dcos_drk_y = (rij_y - dot*rkj_inv*rkj_inv*rkj_y) / tmp;
    scalar_t dcos_drk_z = (rij_z - dot*rkj_inv*rkj_inv*rkj_z) / tmp;

    // NEW: expose dcos/dr so caller can reuse for ba forces without recomputing geometry
    *dcos_dri_x_out = dcos_dri_x; *dcos_dri_y_out = dcos_dri_y; *dcos_dri_z_out = dcos_dri_z;
    *dcos_drk_x_out = dcos_drk_x; *dcos_drk_y_out = dcos_drk_y; *dcos_drk_z_out = dcos_drk_z;

    *dri_x = dV_dcos * dcos_dri_x;
    *dri_y = dV_dcos * dcos_dri_y;
    *dri_z = dV_dcos * dcos_dri_z;
    *drk_x = dV_dcos * dcos_drk_x;
    *drk_y = dV_dcos * dcos_drk_y;
    *drk_z = dV_dcos * dcos_drk_z;

    // --- dV/deps: field projection term ---
    scalar_t dV_deps  = scalar_t(0.0);
    scalar_t ddc_deps = scalar_t(0.0);  // NEW: d(dc)/d(eps), needed for ba efield grad
    if (abs_(eps) > scalar_t(1e-10)) {
        scalar_t eps_inv = scalar_t(1.0) / eps;
        scalar_t dkeff_deps      = clamped ? scalar_t(0.0) : -lambda2 * eps_inv;
        scalar_t dl1_deps        = lambda1 * eps_inv;
        scalar_t dcos_t0eff_deps = dl1_deps / k_eff
                                 - lambda1 / (k_eff * k_eff) * dkeff_deps;

        // ddc_deps = d(cos_theta - cos_t0_eff)/deps = -dcos_t0eff_deps
        // NEW: ba coupling V_ba = k_ba * dr * dc depends on eps through dc,
        // so caller needs ddc_deps to compute dV_ba/dE = k_ba * dr * ddc_deps * d_hat
        ddc_deps = -dcos_t0eff_deps;

        dV_deps = scalar_t(0.5) * dc * dc * dkeff_deps
                + (-k_eff * dc) * dcos_t0eff_deps;

        scalar_t Epd_x = efx - eps*d_x;
        scalar_t Epd_y = efy - eps*d_y;
        scalar_t Epd_z = efz - eps*d_z;
        scalar_t Epd_dot_uij = Epd_x*uij_x + Epd_y*uij_y + Epd_z*uij_z;
        scalar_t deps_dri_x = (Epd_x - Epd_dot_uij*uij_x) * rij_inv * b_norm_inv;
        scalar_t deps_dri_y = (Epd_y - Epd_dot_uij*uij_y) * rij_inv * b_norm_inv;
        scalar_t deps_dri_z = (Epd_z - Epd_dot_uij*uij_z) * rij_inv * b_norm_inv;
        scalar_t Epd_dot_ukj = Epd_x*ukj_x + Epd_y*ukj_y + Epd_z*ukj_z;
        scalar_t deps_drk_x = (Epd_x - Epd_dot_ukj*ukj_x) * rkj_inv * b_norm_inv;
        scalar_t deps_drk_y = (Epd_y - Epd_dot_ukj*ukj_y) * rkj_inv * b_norm_inv;
        scalar_t deps_drk_z = (Epd_z - Epd_dot_ukj*ukj_z) * rkj_inv * b_norm_inv;

        // angle-only field projection contrib to coord grads
        // NOTE: caller adds ba contributions separately using ddc_deps
        *dri_x += dV_deps * deps_dri_x;
        *dri_y += dV_deps * deps_dri_y;
        *dri_z += dV_deps * deps_dri_z;
        *drk_x += dV_deps * deps_drk_x;
        *drk_y += dV_deps * deps_drk_y;
        *drk_z += dV_deps * deps_drk_z;
    }

    // NEW: expose ddc_deps for caller ba efield gradient
    *ddc_deps_out = ddc_deps;

    // angle-only efield grad; caller adds ba contribution
    *def_x = dV_deps * d_x;
    *def_y = dV_deps * d_y;
    *def_z = dV_deps * d_z;
}

#endif // TORCHFF_FD_COSINE_ANGLE_CUH