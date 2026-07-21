#include <torch/autograd.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

#include "angle/fd_cosine_angle.cuh"
#include "common/reduce.cuh"

template <typename scalar_t, int BLOCK_SIZE>
__global__ void cmm_fd_angle_forward_kernel(
    scalar_t* coords,
    int64_t*  angles,
    scalar_t* theta_0,
    scalar_t* k_theta,
    scalar_t* r_eq_1,
    scalar_t* r_eq_2,
    scalar_t* k_bb,
    scalar_t* k_ba_1,
    scalar_t* k_ba_2,
    scalar_t* j_cf_bb,
    scalar_t* j_cf_angle,
    scalar_t* dmu_dtheta,
    scalar_t* d2mu_dtheta2,
    scalar_t* efield,
    scalar_t  ene_coupling_min,
    int64_t   nangles,
    scalar_t* ene_out,
    scalar_t* dq_a,
    scalar_t* coords_grad,
    scalar_t* efield_grad
)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ene_out[0] = scalar_t(0.0);
    }
    __syncthreads();

    scalar_t ene = scalar_t(0.0);

    int64_t start = threadIdx.x + BLOCK_SIZE * blockIdx.x;
    for (int64_t index = start; index < nangles; index += gridDim.x * BLOCK_SIZE) {
        int64_t i = angles[index*3];
        int64_t j = angles[index*3+1];
        int64_t k = angles[index*3+2];

        scalar_t rij_x = coords[i*3]   - coords[j*3];
        scalar_t rij_y = coords[i*3+1] - coords[j*3+1];
        scalar_t rij_z = coords[i*3+2] - coords[j*3+2];
        scalar_t rkj_x = coords[k*3]   - coords[j*3];
        scalar_t rkj_y = coords[k*3+1] - coords[j*3+1];
        scalar_t rkj_z = coords[k*3+2] - coords[j*3+2];

        // Updated call — now receives dc, dcos/dr, ddc_deps, d_hat from fd_cos_angle
        scalar_t e, dc;
        scalar_t dcos_dri_x, dcos_dri_y, dcos_dri_z;
        scalar_t dcos_drk_x, dcos_drk_y, dcos_drk_z;
        scalar_t ddc_deps, d_hat_x, d_hat_y, d_hat_z;
        scalar_t dri_x, dri_y, dri_z, drk_x, drk_y, drk_z, def_x, def_y, def_z;

        fd_cos_angle(
            rij_x, rij_y, rij_z,
            rkj_x, rkj_y, rkj_z,
            efield[j*3], efield[j*3+1], efield[j*3+2],
            theta_0[index], k_theta[index],
            dmu_dtheta[index], d2mu_dtheta2[index],
            &e, &dc,
            &dcos_dri_x, &dcos_dri_y, &dcos_dri_z,
            &dcos_drk_x, &dcos_drk_y, &dcos_drk_z,
            &ddc_deps, &d_hat_x, &d_hat_y, &d_hat_z,
            &dri_x, &dri_y, &dri_z,
            &drk_x, &drk_y, &drk_z,
            &def_x, &def_y, &def_z
        );

        // compute geometry needed for bb/ba and charge flux
        scalar_t rij  = sqrt_(rij_x*rij_x + rij_y*rij_y + rij_z*rij_z);
        scalar_t rkj  = sqrt_(rkj_x*rkj_x + rkj_y*rkj_y + rkj_z*rkj_z);
        scalar_t dot  = rij_x*rkj_x + rij_y*rkj_y + rij_z*rkj_z;
        constexpr scalar_t ONE  = scalar_t(0.9999999999);
        constexpr scalar_t MONE = scalar_t(-0.999999999);
        scalar_t cos_theta = clamp_(dot / (rij * rkj), MONE, ONE);
        scalar_t cos_t0 = cos_(theta_0[index]);
        scalar_t d_cos_theta = cos_theta - cos_t0;  // base delta for bb/ba, same as Python

        scalar_t req1 = r_eq_1[index], req2 = r_eq_2[index];
        scalar_t kbb  = k_bb[index];
        scalar_t kba1 = k_ba_1[index], kba2 = k_ba_2[index];
        scalar_t tmp;

        // bond-angle coupling 1 (bond ij)
        tmp = kba1 * (rij - req1) * d_cos_theta;
        if (tmp > ene_coupling_min) {
            e += tmp;
            dri_x += kba1 * ((rij - req1) * dcos_dri_x + d_cos_theta * rij_x / rij);
            dri_y += kba1 * ((rij - req1) * dcos_dri_y + d_cos_theta * rij_y / rij);
            dri_z += kba1 * ((rij - req1) * dcos_dri_z + d_cos_theta * rij_z / rij);
            drk_x += kba1 * (rij - req1) * dcos_drk_x;
            drk_y += kba1 * (rij - req1) * dcos_drk_y;
            drk_z += kba1 * (rij - req1) * dcos_drk_z;
        } else { e += ene_coupling_min; }

        // bond-angle coupling 2 (bond kj)
        tmp = kba2 * (rkj - req2) * d_cos_theta;
        if (tmp > ene_coupling_min) {
            e += tmp;
            drk_x += kba2 * ((rkj - req2) * dcos_drk_x + d_cos_theta * rkj_x / rkj);
            drk_y += kba2 * ((rkj - req2) * dcos_drk_y + d_cos_theta * rkj_y / rkj);
            drk_z += kba2 * ((rkj - req2) * dcos_drk_z + d_cos_theta * rkj_z / rkj);
            dri_x += kba2 * (rkj - req2) * dcos_dri_x;
            dri_y += kba2 * (rkj - req2) * dcos_dri_y;
            dri_z += kba2 * (rkj - req2) * dcos_dri_z;
        } else { e += ene_coupling_min; }

        // bond-bond coupling
        tmp = kbb * (rij - req1) * (rkj - req2);
        if (tmp > ene_coupling_min) {
            e += tmp;
            tmp = kbb * (rkj - req2) / rij;
            dri_x += tmp * rij_x; dri_y += tmp * rij_y; dri_z += tmp * rij_z;
            tmp = kbb * (rij - req1) / rkj;
            drk_x += tmp * rkj_x; drk_y += tmp * rkj_y; drk_z += tmp * rkj_z;
        } else { e += ene_coupling_min; }

        ene += e;

        // Charge flux — reuses cos_theta already computed above
        scalar_t jcf = j_cf_angle[index];
        scalar_t dtheta = acos_(cos_theta) - theta_0[index];
        scalar_t dqi = jcf * dtheta;
        scalar_t dqk = jcf * dtheta;
        jcf = j_cf_bb[index];
        dqi += jcf * (rkj - req2);
        dqk += jcf * (rij - req1);

        atomicAdd(&dq_a[i], dqi);
        atomicAdd(&dq_a[k], dqk);
        atomicAdd(&dq_a[j], -dqi - dqk);

        atomicAdd(&coords_grad[i*3],   dri_x);
        atomicAdd(&coords_grad[i*3+1], dri_y);
        atomicAdd(&coords_grad[i*3+2], dri_z);
        atomicAdd(&coords_grad[k*3],   drk_x);
        atomicAdd(&coords_grad[k*3+1], drk_y);
        atomicAdd(&coords_grad[k*3+2], drk_z);
        atomicAdd(&coords_grad[j*3],   -dri_x - drk_x);
        atomicAdd(&coords_grad[j*3+1], -dri_y - drk_y);
        atomicAdd(&coords_grad[j*3+2], -dri_z - drk_z);
        atomicAdd(&efield_grad[j*3],   def_x);
        atomicAdd(&efield_grad[j*3+1], def_y);
        atomicAdd(&efield_grad[j*3+2], def_z);
    }
    block_reduce_sum<scalar_t, BLOCK_SIZE>(ene, ene_out);
}
           


// Backward kernel is identical to cmm_angles_backward_kernel —
// handles charge flux gradient w.r.t. coords, unchanged by FD angle
template <typename scalar_t, int BLOCK_SIZE>
__global__ void cmm_fd_angle_backward_kernel(
    scalar_t* coords,
    int64_t*  angles,
    scalar_t* r_eq_1,
    scalar_t* r_eq_2,
    scalar_t* j_cf_angle,
    scalar_t* j_cf_bb,
    int64_t   nangles,
    scalar_t* dq_a_grad,
    scalar_t* coords_grad
)
{
    int64_t start = threadIdx.x + BLOCK_SIZE * blockIdx.x;
    for (int64_t index = start; index < nangles; index += gridDim.x * BLOCK_SIZE) {
        int64_t i = angles[index*3];
        int64_t j = angles[index*3+1];
        int64_t k = angles[index*3+2];

        scalar_t rij_x = coords[i*3]   - coords[j*3];
        scalar_t rij_y = coords[i*3+1] - coords[j*3+1];
        scalar_t rij_z = coords[i*3+2] - coords[j*3+2];
        scalar_t rkj_x = coords[k*3]   - coords[j*3];
        scalar_t rkj_y = coords[k*3+1] - coords[j*3+1];
        scalar_t rkj_z = coords[k*3+2] - coords[j*3+2];

        scalar_t dot = rij_x*rkj_x + rij_y*rkj_y + rij_z*rkj_z;
        scalar_t rij = sqrt_(rij_x*rij_x + rij_y*rij_y + rij_z*rij_z);
        scalar_t rkj = sqrt_(rkj_x*rkj_x + rkj_y*rkj_y + rkj_z*rkj_z);

        constexpr scalar_t ONE  = scalar_t(0.9999999999);
        constexpr scalar_t MONE = scalar_t(-0.999999999);
        scalar_t tmp = clamp_(dot / rij / rkj, MONE, ONE);
        scalar_t dtheta_dcos = -scalar_t(1.0) / sqrt_(scalar_t(1.0) - tmp*tmp);
        tmp = dtheta_dcos / rij / rkj;

        scalar_t dtheta_dri_x = (rkj_x - dot/rij/rij*rij_x) * tmp;
        scalar_t dtheta_dri_y = (rkj_y - dot/rij/rij*rij_y) * tmp;
        scalar_t dtheta_dri_z = (rkj_z - dot/rij/rij*rij_z) * tmp;
        scalar_t dtheta_drk_x = (rij_x - dot/rkj/rkj*rkj_x) * tmp;
        scalar_t dtheta_drk_y = (rij_y - dot/rkj/rkj*rkj_y) * tmp;
        scalar_t dtheta_drk_z = (rij_z - dot/rkj/rkj*rkj_z) * tmp;

        scalar_t dqi_grad = dq_a_grad[i];
        scalar_t dqj_grad = dq_a_grad[j];
        scalar_t dqk_grad = dq_a_grad[k];

        scalar_t jcf_bb = j_cf_bb[index];
        tmp = (dqi_grad - 2*dqj_grad + dqk_grad) * j_cf_angle[index];
        scalar_t tmp2 = (dqk_grad - dqj_grad) * jcf_bb / rij;

        scalar_t dri_x = tmp*dtheta_dri_x + tmp2*rij_x;
        scalar_t dri_y = tmp*dtheta_dri_y + tmp2*rij_y;
        scalar_t dri_z = tmp*dtheta_dri_z + tmp2*rij_z;
        tmp2 = (dqi_grad - dqj_grad) * jcf_bb / rkj;
        scalar_t drk_x = tmp*dtheta_drk_x + tmp2*rkj_x;
        scalar_t drk_y = tmp*dtheta_drk_y + tmp2*rkj_y;
        scalar_t drk_z = tmp*dtheta_drk_z + tmp2*rkj_z;

        atomicAdd(&coords_grad[i*3],   dri_x);
        atomicAdd(&coords_grad[i*3+1], dri_y);
        atomicAdd(&coords_grad[i*3+2], dri_z);
        atomicAdd(&coords_grad[k*3],   drk_x);
        atomicAdd(&coords_grad[k*3+1], drk_y);
        atomicAdd(&coords_grad[k*3+2], drk_z);
        atomicAdd(&coords_grad[j*3],   -dri_x - drk_x);
        atomicAdd(&coords_grad[j*3+1], -dri_y - drk_y);
        atomicAdd(&coords_grad[j*3+2], -dri_z - drk_z);
    }
}


class CMMFDAngleCuda : public torch::autograd::Function<CMMFDAngleCuda> {
public:
    static std::vector<at::Tensor> forward(
        torch::autograd::AutogradContext* ctx,
        at::Tensor& coords, at::Tensor& angles,
        at::Tensor& theta_0, at::Tensor& k_theta,
        at::Tensor& r_eq_1, at::Tensor& r_eq_2,
        at::Tensor& k_bb, at::Tensor& k_ba_1, at::Tensor& k_ba_2,
        at::Tensor& j_cf_bb, at::Tensor& j_cf_angle,
        at::Tensor& dmu_dtheta, at::Tensor& d2mu_dtheta2,
        at::Tensor& efield,
        at::Scalar ene_coupling_min
    ) {
        int64_t natoms  = coords.size(0);
        int64_t nangles = angles.size(0);

        auto opts = coords.options();
        at::Tensor coords_grad = at::zeros_like(coords, opts);
        at::Tensor efield_grad = at::zeros_like(efield, opts);
        at::Tensor ene  = at::zeros({}, opts);
        at::Tensor dq_a = at::zeros({natoms}, opts);

        auto props = at::cuda::getCurrentDeviceProperties();
        auto stream = at::cuda::getCurrentCUDAStream();
        constexpr int BLOCK_SIZE = 256;
        int64_t grid_dim = std::min(
            static_cast<int64_t>(props->maxBlocksPerMultiProcessor * props->multiProcessorCount),
            (nangles + BLOCK_SIZE - 1) / BLOCK_SIZE);

        AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "cmm_fd_angle_forward_kernel", ([&] {
            cmm_fd_angle_forward_kernel<scalar_t, BLOCK_SIZE><<<grid_dim, BLOCK_SIZE, 0, stream>>>(
                coords.data_ptr<scalar_t>(),
                angles.data_ptr<int64_t>(),
                theta_0.data_ptr<scalar_t>(),
                k_theta.data_ptr<scalar_t>(),
                r_eq_1.data_ptr<scalar_t>(),
                r_eq_2.data_ptr<scalar_t>(),
                k_bb.data_ptr<scalar_t>(),
                k_ba_1.data_ptr<scalar_t>(),
                k_ba_2.data_ptr<scalar_t>(),
                j_cf_bb.data_ptr<scalar_t>(),
                j_cf_angle.data_ptr<scalar_t>(),
                dmu_dtheta.data_ptr<scalar_t>(),
                d2mu_dtheta2.data_ptr<scalar_t>(),
                efield.data_ptr<scalar_t>(),
                static_cast<scalar_t>(ene_coupling_min.toDouble()),
                nangles,
                ene.data_ptr<scalar_t>(),
                dq_a.data_ptr<scalar_t>(),
                coords_grad.data_ptr<scalar_t>(),
                efield_grad.data_ptr<scalar_t>()
            );
        }));

        ctx->save_for_backward({
            coords_grad, efield_grad,
            coords, angles,
            r_eq_1, r_eq_2, j_cf_angle, j_cf_bb
        });
        ctx->saved_data["nangles"] = nangles;

        return {ene, dq_a};
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        std::vector<at::Tensor> grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        at::Tensor coords_grad = saved[0] * grad_outputs[0];
        at::Tensor efield_grad = saved[1] * grad_outputs[0];

        int64_t nangles = static_cast<int64_t>(ctx->saved_data["nangles"].toInt());

        auto props = at::cuda::getCurrentDeviceProperties();
        auto stream = at::cuda::getCurrentCUDAStream();
        constexpr int BLOCK_SIZE = 256;
        int64_t grid_dim = std::min(
            static_cast<int64_t>(props->maxBlocksPerMultiProcessor * props->multiProcessorCount),
            (nangles + BLOCK_SIZE - 1) / BLOCK_SIZE);

        AT_DISPATCH_FLOATING_TYPES(saved[0].scalar_type(), "cmm_fd_angle_backward_kernel", ([&] {
            cmm_fd_angle_backward_kernel<scalar_t, BLOCK_SIZE><<<grid_dim, BLOCK_SIZE, 0, stream>>>(
                saved[2].data_ptr<scalar_t>(),  // coords
                saved[3].data_ptr<int64_t>(),   // angles
                saved[4].data_ptr<scalar_t>(),  // r_eq_1
                saved[5].data_ptr<scalar_t>(),  // r_eq_2
                saved[6].data_ptr<scalar_t>(),  // j_cf_angle
                saved[7].data_ptr<scalar_t>(),  // j_cf_bb
                nangles,
                grad_outputs[1].contiguous().data_ptr<scalar_t>(),
                coords_grad.data_ptr<scalar_t>()
            );
        }));

        at::Tensor ignore;
        return {
            coords_grad,  // coords
            ignore,       // angles
            ignore,       // theta_0
            ignore,       // k_theta
            ignore,       // r_eq_1
            ignore,       // r_eq_2
            ignore,       // k_bb
            ignore,       // k_ba_1
            ignore,       // k_ba_2
            ignore,       // j_cf_bb
            ignore,       // j_cf_angle
            ignore,       // dmu_dtheta
            ignore,       // d2mu_dtheta2
            efield_grad,  // efield
            ignore        // ene_coupling_min
        };
    }
};


std::tuple<at::Tensor, at::Tensor> cmm_fd_angle_cuda(
    at::Tensor& coords, at::Tensor& angles,
    at::Tensor& theta_0, at::Tensor& k_theta,
    at::Tensor& r_eq_1, at::Tensor& r_eq_2,
    at::Tensor& k_bb, at::Tensor& k_ba_1, at::Tensor& k_ba_2,
    at::Tensor& j_cf_bb, at::Tensor& j_cf_angle,
    at::Tensor& dmu_dtheta, at::Tensor& d2mu_dtheta2,
    at::Tensor& efield,
    at::Scalar ene_coupling_min
) {
    auto outs = CMMFDAngleCuda::apply(
        coords, angles,
        theta_0, k_theta,
        r_eq_1, r_eq_2,
        k_bb, k_ba_1, k_ba_2,
        j_cf_bb, j_cf_angle,
        dmu_dtheta, d2mu_dtheta2,
        efield,
        ene_coupling_min
    );
    return {outs[0], outs[1]};
}

TORCH_LIBRARY_IMPL(torchff, AutogradCUDA, m) {
    m.impl("cmm_fd_angle", cmm_fd_angle_cuda);
}