import os
import math
import torch
from torch import Tensor
from typing import Dict, Optional, Tuple, Literal, Union

# Import CUDA functions from our compiled extension if available
try:
    import surfel_rasterizer_extension
except ImportError:
    surfel_rasterizer_extension = None

from gsplat.rendering import (
    RenderMode,
    render_mode_has_color,
    render_mode_has_depth,
    render_mode_has_expected_depth,
    render_mode_has_depth_channel,
    render_mode_has_only_depth_channel,
)
from gsplat.cuda._wrapper import (
    fully_fused_projection_2dgs,
    isect_offset_encode,
    isect_tiles,
    spherical_harmonics,
)
from gsplat.utils import depth_to_normal

class _RasterizeToPixelsSurfel(torch.autograd.Function):
    """Wrapper autograd function calling compiled CUDA kernels for surfels."""

    @staticmethod
    def forward(
        ctx,
        means2d: Tensor,
        ray_transforms: Tensor,
        colors: Tensor,
        opacities: Tensor,
        normals: Tensor,
        densify: Tensor,
        backgrounds: Optional[Tensor],
        masks: Optional[Tensor],
        width: int,
        height: int,
        tile_size: int,
        isect_offsets: Tensor,
        flatten_ids: Tensor,
        absgrad: bool,
        distloss: bool,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if surfel_rasterizer_extension is None:
            raise RuntimeError(
                "surfel_rasterizer_extension is not compiled. Please set CUDA_HOME and install it "
                "using `pip install -e .` in `external_code/render`."
            )
        (
            render_colors,
            render_alphas,
            render_normals,
            render_distort,
            render_median,
            last_ids,
            median_ids,
        ) = surfel_rasterizer_extension.rasterize_to_pixels_surfel_fwd(
            means2d,
            ray_transforms,
            colors,
            opacities,
            normals,
            backgrounds,
            masks,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
        )

        ctx.save_for_backward(
            means2d,
            ray_transforms,
            colors,
            opacities,
            normals,
            densify,
            backgrounds,
            masks,
            isect_offsets,
            flatten_ids,
            render_colors,
            render_alphas,
            last_ids,
            median_ids,
        )
        ctx.width = width
        ctx.height = height
        ctx.tile_size = tile_size
        ctx.absgrad = absgrad
        ctx.distloss = distloss

        # double to float
        render_alphas = render_alphas.float()
        return (
            render_colors,
            render_alphas,
            render_normals,
            render_distort,
            render_median,
        )

    @staticmethod
    def backward(
        ctx,
        v_render_colors: Tensor,
        v_render_alphas: Tensor,
        v_render_normals: Tensor,
        v_render_distort: Tensor,
        v_render_median: Tensor,
    ):
        (
            means2d,
            ray_transforms,
            colors,
            opacities,
            normals,
            densify,
            backgrounds,
            masks,
            isect_offsets,
            flatten_ids,
            render_colors,
            render_alphas,
            last_ids,
            median_ids,
        ) = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        tile_size = ctx.tile_size
        absgrad = ctx.absgrad

        if surfel_rasterizer_extension is None:
            raise RuntimeError(
                "surfel_rasterizer_extension is not compiled. Please set CUDA_HOME and install it "
                "using `pip install -e .` in `external_code/render`."
            )
        (
            v_means2d_abs,
            v_means2d,
            v_ray_transforms,
            v_colors,
            v_opacities,
            v_normals,
            v_densify,
        ) = surfel_rasterizer_extension.rasterize_to_pixels_surfel_bwd(
            means2d,
            ray_transforms,
            colors,
            opacities,
            normals,
            densify,
            backgrounds,
            masks,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            render_colors,
            render_alphas,
            last_ids,
            median_ids,
            v_render_colors.contiguous(),
            v_render_alphas.contiguous(),
            v_render_normals.contiguous(),
            v_render_distort.contiguous(),
            v_render_median.contiguous(),
            absgrad,
        )
        torch.cuda.synchronize()
        if absgrad:
            means2d.absgrad = v_means2d_abs

        if ctx.needs_input_grad[6]:
            v_backgrounds = (v_render_colors * (1.0 - render_alphas).float()).sum(
                dim=(-3, -2)
            )
        else:
            v_backgrounds = None

        return (
            v_means2d,
            v_ray_transforms,
            v_colors,
            v_opacities,
            v_normals,
            v_densify,
            v_backgrounds,
            None,  # masks
            None,  # width
            None,  # height
            None,  # tile_size
            None,  # isect_offsets
            None,  # flatten_ids
            None,  # absgrad
            None,  # distloss
        )


def rasterize_to_pixels_surfel(
    means2d: Tensor,  # [..., N, 2]
    ray_transforms: Tensor,  # [..., N, 3, 3]
    colors: Tensor,  # [..., N, channels]
    opacities: Tensor,  # [..., N]
    normals: Tensor,  # [..., N, 3]
    densify: Tensor,  # [..., N, 2]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [..., tile_height, tile_width]
    flatten_ids: Tensor,  # [n_isects]
    backgrounds: Optional[Tensor] = None,  # [..., channels]
    masks: Optional[Tensor] = None,  # [..., tile_height, tile_width]
    packed: bool = False,
    absgrad: bool = False,
    distloss: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Rasterize Surfels to pixels using PyTorch autograd Function wrapper."""
    image_dims = means2d.shape[:-2]
    channels = colors.shape[-1]
    device = means2d.device
    if packed:
        nnz = means2d.size(0)
        assert means2d.shape == (nnz, 2), means2d.shape
        assert ray_transforms.shape == (nnz, 3, 3), ray_transforms.shape
        assert colors.shape[0] == nnz, colors.shape
        assert opacities.shape == (nnz,), opacities.shape
    else:
        N = means2d.size(-2)
        assert means2d.shape == image_dims + (N, 2), means2d.shape
        assert ray_transforms.shape == image_dims + (N, 3, 3), ray_transforms.shape
        assert colors.shape[:-2] == image_dims, colors.shape
        assert opacities.shape == image_dims + (N,), opacities.shape
    if backgrounds is not None:
        assert backgrounds.shape == image_dims + (channels,), backgrounds.shape
        backgrounds = backgrounds.contiguous()

    # Pad the channels to the nearest supported number if necessary
    if channels > 512 or channels == 0:
        raise ValueError(f"Unsupported number of color channels: {channels}")
    if channels not in (1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512):
        padded_channels = (1 << (channels - 1).bit_length()) - channels
        # Make sure the depth (last channel if present) remains in the last channel after padding
        colors = torch.cat(
            [
                colors[..., :-1],
                torch.empty(*colors.shape[:-1], padded_channels, device=device),
                colors[..., -1:],
            ],
            dim=-1,
        )
        if backgrounds is not None:
            backgrounds = torch.cat(
                [
                    backgrounds,
                    torch.zeros(
                        *backgrounds.shape[:-1], padded_channels, device=device
                    ),
                ],
                dim=-1,
            )
    else:
        padded_channels = 0
    tile_height, tile_width = isect_offsets.shape[-2:]
    assert (
        tile_height * tile_size >= image_height
    ), f"Assert Failed: {tile_height} * {tile_size} >= {image_height}"
    assert (
        tile_width * tile_size >= image_width
    ), f"Assert Failed: {tile_width} * {tile_size} >= {image_width}"

    (
        render_colors,
        render_alphas,
        render_normals,
        render_distort,
        render_median,
    ) = _RasterizeToPixelsSurfel.apply(
        means2d.contiguous(),
        ray_transforms.contiguous(),
        colors.contiguous(),
        opacities.contiguous(),
        normals.contiguous(),
        densify.contiguous(),
        backgrounds,
        masks,
        image_width,
        image_height,
        tile_size,
        isect_offsets.contiguous(),
        flatten_ids.contiguous(),
        absgrad,
        distloss,
    )

    if padded_channels > 0:
        render_colors = torch.cat(
            [render_colors[..., : -padded_channels - 1], render_colors[..., -1:]],
            dim=-1,
        )

    return render_colors, render_alphas, render_normals, render_distort, render_median


def rasterization_surfel(
    means: Tensor,  # [..., N, 3]
    quats: Tensor,  # [..., N, 4]
    scales: Tensor,  # [..., N, 3]
    opacities: Tensor,  # [..., N]
    colors: Tensor,  # [..., (C,) N, D] or [..., (C,) N, K, 3]
    viewmats: Tensor,  # [..., C, 4, 4]
    Ks: Tensor,  # [..., C, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    sh_degree: Optional[int] = None,
    packed: bool = False,
    tile_size: int = 16,
    backgrounds: Optional[Tensor] = None,
    render_mode: RenderMode = "RGB",
    sparse_grad: bool = False,
    absgrad: bool = False,
    distloss: bool = False,
    depth_mode: Literal["expected", "median"] = "expected",
) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Tensor, Tensor, Dict]:
    """High-level Python wrapper for surfel rasterization, matching `rasterization_2dgs`."""
    batch_dims = means.shape[:-2]
    num_batch_dims = len(batch_dims)
    B = math.prod(batch_dims)
    N = means.shape[-2]
    C = viewmats.shape[-3]
    I = B * C
    device = means.device
    channels = colors.shape[-1]

    # Added proper assertions matching gsplat
    assert means.shape == batch_dims + (N, 3), means.shape
    assert quats.shape == batch_dims + (N, 4), quats.shape
    assert scales.shape == batch_dims + (N, 3), scales.shape
    assert opacities.shape == batch_dims + (N,), opacities.shape
    assert viewmats.shape == batch_dims + (C, 4, 4), viewmats.shape
    assert Ks.shape == batch_dims + (C, 3, 3), Ks.shape
    if distloss:
        assert render_mode_has_depth(
            render_mode
        ), f"distloss requires depth rendering, but render mode is {render_mode}"

    if sh_degree is None:
        # treat colors as post-activation values, should be in shape [..., N, D] or [..., C, N, D]
        assert (
            colors.dim() == num_batch_dims + 2
            and colors.shape[:-1] == batch_dims + (N,)
        ) or (
            colors.dim() == num_batch_dims + 3
            and colors.shape[:-1] == batch_dims + (C, N)
        ), colors.shape
    else:
        # treat colors as SH coefficients, should be in shape [..., N, K, 3] or [..., C, N, K, 3]
        assert (
            colors.dim() == num_batch_dims + 3
            and colors.shape[:-2] == batch_dims + (N,)
            and colors.shape[-1] == 3
        ) or (
            colors.dim() == num_batch_dims + 4
            and colors.shape[:-2] == batch_dims + (C, N)
            and colors.shape[-1] == 3
        ), colors.shape
        assert (sh_degree + 1) ** 2 <= colors.shape[-2], colors.shape

    # Compute ray-surfel projections using standard 2DGS EWA projection from gsplat
    proj_results = fully_fused_projection_2dgs(
        means,
        quats,
        scales,
        viewmats,
        Ks,
        width,
        height,
        eps2d,
        near_plane,
        far_plane,
        radius_clip,
        packed,
        sparse_grad,
    )

    if packed:
        (
            batch_ids,
            camera_ids,
            gaussian_ids,
            radii,
            means2d,
            depths,
            ray_transforms,
            normals,
        ) = proj_results
        opacities = opacities.view(B, N)[batch_ids, gaussian_ids]
        image_ids = batch_ids * C + camera_ids
    else:
        radii, means2d, depths, ray_transforms, normals = proj_results
        opacities = torch.broadcast_to(
            opacities[..., None, :], batch_dims + (C, N)
        )
        camera_ids, gaussian_ids = None, None
        image_ids = None

    densify = torch.zeros_like(
        means2d, dtype=means.dtype, requires_grad=True, device="cuda"
    )

    # Intersection offsets and tiles
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        packed=packed,
        n_images=I,
        image_ids=image_ids,
        gaussian_ids=gaussian_ids,
    )
    isect_offsets = isect_offset_encode(isect_ids, I, tile_width, tile_height)
    isect_offsets = isect_offsets.reshape(batch_dims + (C, tile_height, tile_width))

    # Per-view colors (SH)
    if sh_degree is not None:
        camtoworlds = torch.inverse(viewmats)
        if packed:
            dirs = means[..., gaussian_ids, :] - camtoworlds[..., camera_ids, :3, 3]
        else:
            dirs = means[..., None, :, :] - camtoworlds[..., None, :3, 3]

        if colors.dim() == num_batch_dims + 3:
            shs = torch.broadcast_to(
                colors[..., None, :, :, :], batch_dims + (C, N, -1, 3)
            )
        else:
            shs = colors
        colors = spherical_harmonics(
            sh_degree, dirs, shs, masks=(radii > 0).all(dim=-1)
        )
        colors = torch.clamp_min(colors + 0.5, 0.0)

    # Render mode depth handling
    if render_mode_has_depth_channel(render_mode) and render_mode_has_color(render_mode):
        colors = torch.cat((colors, depths[..., None]), dim=-1)
        if backgrounds is not None:
            backgrounds = torch.cat(
                (backgrounds, torch.zeros_like(backgrounds[..., :1])), dim=-1
            )
    elif render_mode_has_only_depth_channel(render_mode):
        colors = depths[..., None]

    # Call custom autograd function wrapper
    (
        render_colors,
        render_alphas,
        render_normals,
        render_distort,
        render_median,
    ) = rasterize_to_pixels_surfel(
        means2d,
        ray_transforms,
        colors,
        opacities,
        normals,
        densify,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds,
        packed=packed,
        absgrad=absgrad,
        distloss=distloss,
    )

    # Post-process expected depth if requested
    render_normals_from_depth = None
    if render_mode_has_expected_depth(render_mode):
        render_colors = torch.cat(
            [
                render_colors[..., :-1],
                render_colors[..., -1:] / render_alphas.clamp(min=1e-10),
            ],
            dim=-1,
        )
    if render_mode_has_depth(render_mode) and render_mode_has_color(render_mode):
        if depth_mode == "expected":
            depth_for_normal = render_colors[..., -1:]
        elif depth_mode == "median":
            depth_for_normal = render_median

        render_normals_from_depth = depth_to_normal(
            depth_for_normal, torch.linalg.inv(viewmats), Ks
        ).squeeze(0)

    meta = {
        "camera_ids": camera_ids,
        "gaussian_ids": gaussian_ids,
        "radii": radii,
        "means2d": means2d,
        "depths": depths,
        "ray_transforms": ray_transforms,
        "opacities": opacities,
        "normals": normals,
        "tile_width": tile_width,
        "tile_height": tile_height,
        "tiles_per_gauss": tiles_per_gauss,
        "isect_ids": isect_ids,
        "flatten_ids": flatten_ids,
        "isect_offsets": isect_offsets,
        "width": width,
        "height": height,
        "tile_size": tile_size,
        "n_cameras": C,
        "render_distort": render_distort,
        "gradient_2dgs": densify,
    }

    return (
        render_colors,
        render_alphas,
        render_normals,
        render_normals_from_depth,
        render_distort,
        render_median,
        meta,
    )
