#include <torch/extension.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAGuard.h>
#include <tuple>

#include "Common.h"
#include "Config.h"
#include "MacroUtils.h"

namespace gsplat {

template <uint32_t CDIM>
void launch_rasterize_to_pixels_surfel_fwd_kernel(
    // Gaussian parameters
    const at::Tensor means2d,        // [..., N, 2] or [nnz, 2]
    const at::Tensor ray_transforms, // [..., N, 3, 3] or [nnz, 3, 3]
    const at::Tensor colors,         // [..., N, channels] or [nnz, channels]
    const at::Tensor opacities,      // [..., N]  or [nnz]
    const at::Tensor normals,        // [..., N, 3]
    const at::optional<at::Tensor> backgrounds, // [..., channels]
    const at::optional<at::Tensor> masks,       // [..., tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    // outputs
    at::Tensor renders,        // [..., image_height, image_width, channels]
    at::Tensor alphas,         // [..., image_height, image_width, 1]
    at::Tensor render_normals, // [..., image_height, image_width, 3]
    at::Tensor render_distort, // [..., image_height, image_width, 1]
    at::Tensor render_median,  // [..., image_height, image_width, 1]
    at::Tensor last_ids,       // [..., image_height, image_width]
    at::Tensor median_ids      // [..., image_height, image_width]
);

template <uint32_t CDIM>
void launch_rasterize_to_pixels_surfel_bwd_kernel(
    // Gaussian parameters
    const at::Tensor means2d,                   // [..., N, 2] or [nnz, 2]
    const at::Tensor ray_transforms,            // [..., N, 3, 3] or [nnz, 3, 3]
    const at::Tensor colors,                    // [..., N, 3] or [nnz, 3]
    const at::Tensor opacities,                 // [..., N] or [nnz]
    const at::Tensor normals,                   // [..., N, 3] or [nnz, 3]
    const at::Tensor densify,                   // [..., N, 2] or [nnz, 2]
    const at::optional<at::Tensor> backgrounds, // [..., 3]
    const at::optional<at::Tensor> masks,       // [..., tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const at::Tensor tile_offsets,
    const at::Tensor flatten_ids,
    const at::Tensor render_colors,
    const at::Tensor render_alphas,
    const at::Tensor last_ids,
    const at::Tensor median_ids,
    const at::Tensor v_render_colors,
    const at::Tensor v_render_alphas,
    const at::Tensor v_render_normals,
    const at::Tensor v_render_distort,
    const at::Tensor v_render_median,
    at::optional<at::Tensor> v_means2d_abs,
    const at::Tensor v_means2d,
    const at::Tensor v_ray_transforms,
    const at::Tensor v_colors,
    const at::Tensor v_opacities,
    const at::Tensor v_normals,
    const at::Tensor v_densify
);

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
rasterize_to_pixels_surfel_fwd(
    // Gaussian parameters
    const at::Tensor &means2d,        // [..., N, 2] or [nnz, 2]
    const at::Tensor &ray_transforms, // [..., N, 3, 3] or [nnz, 3, 3]
    const at::Tensor &colors,         // [..., N, channels] or [nnz, channels]
    const at::Tensor &opacities,      // [..., N]  or [nnz]
    const at::Tensor &normals,        // [..., N, 3] or [nnz, 3]
    const at::optional<at::Tensor> &backgrounds, // [..., channels]
    const at::optional<at::Tensor> &masks,       // [..., tile_height, tile_width]
    // image size
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    // intersections
    const at::Tensor &tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor &flatten_ids   // [n_isects]
) {
    DEVICE_GUARD(means2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(ray_transforms);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacities);
    CHECK_INPUT(normals);
    CHECK_INPUT(tile_offsets);
    CHECK_INPUT(flatten_ids);
    if (backgrounds.has_value()) {
        CHECK_INPUT(backgrounds.value());
    }
    if (masks.has_value()) {
        CHECK_INPUT(masks.value());
    }
    auto opt = means2d.options();

    at::DimVector image_dims(tile_offsets.sizes().slice(0, tile_offsets.dim() - 2));
    uint32_t channels = colors.size(-1);

    at::DimVector renders_dims(image_dims);
    renders_dims.append({image_height, image_width, channels});
    at::Tensor renders = at::empty(renders_dims, opt);

    at::DimVector alphas_dims(image_dims);
    alphas_dims.append({image_height, image_width, 1});
    at::Tensor alphas = at::empty(alphas_dims, opt);

    at::DimVector last_ids_dims(image_dims);
    last_ids_dims.append({image_height, image_width});
    at::Tensor last_ids = at::empty(last_ids_dims, opt.dtype(at::kInt));

    at::DimVector median_ids_dims(image_dims);
    median_ids_dims.append({image_height, image_width});
    at::Tensor median_ids = at::empty(median_ids_dims, opt.dtype(at::kInt));

    at::DimVector render_normals_dims(image_dims);
    render_normals_dims.append({image_height, image_width, 3});
    at::Tensor render_normals = at::empty(render_normals_dims, opt);

    at::DimVector render_distort_dims(image_dims);
    render_distort_dims.append({image_height, image_width, 1});
    at::Tensor render_distort = at::empty(render_distort_dims, opt);

    at::DimVector render_median_dims(image_dims);
    render_median_dims.append({image_height, image_width, 1});
    at::Tensor render_median = at::empty(render_median_dims, opt);

#define __LAUNCH_KERNEL__(N)                                                   \
    case N:                                                                    \
        launch_rasterize_to_pixels_surfel_fwd_kernel<N>(                       \
            means2d,                                                           \
            ray_transforms,                                                    \
            colors,                                                            \
            opacities,                                                         \
            normals,                                                           \
            backgrounds,                                                       \
            masks,                                                             \
            image_width,                                                       \
            image_height,                                                      \
            tile_size,                                                         \
            tile_offsets,                                                      \
            flatten_ids,                                                       \
            renders,                                                           \
            alphas,                                                            \
            render_normals,                                                    \
            render_distort,                                                    \
            render_median,                                                     \
            last_ids,                                                          \
            median_ids                                                         \
        );                                                                     \
        break;

    switch (channels) {
        GSPLAT_FOR_EACH(__LAUNCH_KERNEL__, GSPLAT_NUM_CHANNELS)
    default:
        AT_ERROR("Unsupported number of channels: ", channels);
    }
#undef __LAUNCH_KERNEL__

    return std::make_tuple(
        renders,
        alphas,
        render_normals,
        render_distort,
        render_median,
        last_ids,
        median_ids
    );
}

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
rasterize_to_pixels_surfel_bwd(
    // Gaussian parameters
    const at::Tensor &means2d,        // [..., N, 2] or [nnz, 2]
    const at::Tensor &ray_transforms, // [..., N, 3, 3] or [nnz, 3, 3]
    const at::Tensor &colors,         // [..., N, channels] or [nnz, channels]
    const at::Tensor &opacities,      // [..., N] or [nnz]
    const at::Tensor &normals,        // [..., N, 3] or [nnz, 3]
    const at::Tensor &densify,
    const at::optional<at::Tensor> &backgrounds, // [..., channels]
    const at::optional<at::Tensor> &masks,       // [..., tile_height, tile_width]
    // image size
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    // ray_crossions
    const at::Tensor &tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor &flatten_ids,  // [n_isects]
    // forward outputs
    const at::Tensor &render_colors, // [..., image_height, image_width, channels]
    const at::Tensor &render_alphas, // [..., image_height, image_width, 1]
    const at::Tensor &last_ids,      // [..., image_height, image_width]
    const at::Tensor &median_ids,    // [..., image_height, image_width]
    // gradients of outputs
    const at::Tensor &v_render_colors,  // [..., image_height, image_width, channels]
    const at::Tensor &v_render_alphas,  // [..., image_height, image_width, 1]
    const at::Tensor &v_render_normals, // [..., image_height, image_width, 3]
    const at::Tensor &v_render_distort, // [..., image_height, image_width, 1]
    const at::Tensor &v_render_median,  // [..., image_height, image_width, 1]
    // options
    bool absgrad
) {
    DEVICE_GUARD(means2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(ray_transforms);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacities);
    CHECK_INPUT(normals);
    CHECK_INPUT(densify);
    CHECK_INPUT(tile_offsets);
    CHECK_INPUT(flatten_ids);
    CHECK_INPUT(render_colors);
    CHECK_INPUT(render_alphas);
    CHECK_INPUT(last_ids);
    CHECK_INPUT(median_ids);
    CHECK_INPUT(v_render_colors);
    CHECK_INPUT(v_render_alphas);
    CHECK_INPUT(v_render_normals);
    CHECK_INPUT(v_render_distort);
    CHECK_INPUT(v_render_median);
    if (backgrounds.has_value()) {
        CHECK_INPUT(backgrounds.value());
    }
    if (masks.has_value()) {
        CHECK_INPUT(masks.value());
    }

    uint32_t channels = colors.size(-1);

    at::Tensor v_means2d = at::zeros_like(means2d);
    at::Tensor v_ray_transforms = at::zeros_like(ray_transforms);
    at::Tensor v_colors = at::zeros_like(colors);
    at::Tensor v_normals = at::zeros_like(normals);
    at::Tensor v_opacities = at::zeros_like(opacities);
    at::Tensor v_means2d_abs;
    if (absgrad) {
        v_means2d_abs = at::zeros_like(means2d);
    }
    at::Tensor v_densify = at::zeros_like(densify);

#define __LAUNCH_KERNEL__(N)                                                   \
    case N:                                                                    \
        launch_rasterize_to_pixels_surfel_bwd_kernel<N>(                       \
            means2d,                                                           \
            ray_transforms,                                                    \
            colors,                                                            \
            opacities,                                                         \
            normals,                                                           \
            densify,                                                           \
            backgrounds,                                                       \
            masks,                                                             \
            image_width,                                                       \
            image_height,                                                      \
            tile_size,                                                         \
            tile_offsets,                                                      \
            flatten_ids,                                                       \
            render_colors,                                                     \
            render_alphas,                                                     \
            last_ids,                                                          \
            median_ids,                                                        \
            v_render_colors,                                                   \
            v_render_alphas,                                                   \
            v_render_normals,                                                  \
            v_render_distort,                                                  \
            v_render_median,                                                   \
            absgrad ? at::optional<at::Tensor>(v_means2d_abs) : at::nullopt,   \
            v_means2d,                                                         \
            v_ray_transforms,                                                  \
            v_colors,                                                          \
            v_opacities,                                                       \
            v_normals,                                                         \
            v_densify                                                          \
        );                                                                     \
        break;

    switch (channels) {
        GSPLAT_FOR_EACH(__LAUNCH_KERNEL__, GSPLAT_NUM_CHANNELS)
    default:
        AT_ERROR("Unsupported number of channels: ", channels);
    }
#undef __LAUNCH_KERNEL__

    return std::make_tuple(
        v_means2d_abs,
        v_means2d,
        v_ray_transforms,
        v_colors,
        v_opacities,
        v_normals,
        v_densify
    );
}

} // namespace gsplat

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rasterize_to_pixels_surfel_fwd", &gsplat::rasterize_to_pixels_surfel_fwd);
    m.def("rasterize_to_pixels_surfel_bwd", &gsplat::rasterize_to_pixels_surfel_bwd);
}
