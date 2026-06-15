import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

render_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(render_dir))

include_dirs = [
    os.path.join(project_dir, "external_code", "gsplat", "gsplat", "cuda", "include"),
    os.path.join(project_dir, "external_code", "gsplat", "gsplat", "cuda", "csrc"),
    os.path.join(project_dir, "external_code", "gsplat", "gsplat", "cuda", "csrc", "third_party", "glm"),
]

extra_cflags = ["/std:c++20", "/Zc:preprocessor", "-DWIN32_LEAN_AND_MEAN"]
extra_cuda_cflags = [
    "-std=c++20",
    "-allow-unsupported-compiler",
    "-Xcompiler",
    "/Zc:preprocessor",
    "-DWIN32_LEAN_AND_MEAN",
    "--forward-unknown-opts",
    "-use_fast_math",
    "-diag-suppress",
    "20012,186",
]

setup(
    name="surfel_rasterizer_extension",
    ext_modules=[
        CUDAExtension(
            name="surfel_rasterizer_extension",
            sources=[
                "surfel_rasterizer.cpp",
                "RasterizeToPixelsSurfelFwd.cu",
                "RasterizeToPixelsSurfelBwd.cu",
            ],
            include_dirs=include_dirs,
            extra_compile_args={
                "cxx": extra_cflags,
                "nvcc": extra_cuda_cflags,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
