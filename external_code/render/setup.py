import os
import sys
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Set a fallback TORCH_CUDA_ARCH_LIST if not present to avoid PyTorch IndexError on GPU-less nodes
if "TORCH_CUDA_ARCH_LIST" not in os.environ:
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0;7.5;8.0;8.6;8.9;9.0+PTX"

render_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(render_dir))

include_dirs = [
    os.path.join(project_dir, "external_code", "gsplat", "gsplat", "cuda", "include"),
    os.path.join(project_dir, "external_code", "gsplat", "gsplat", "cuda", "csrc"),
    os.path.join(project_dir, "external_code", "gsplat", "gsplat", "cuda", "csrc", "third_party", "glm"),
]

# Compiler flags ----------------------------------
extra_cflags = []
extra_cuda_cflags = []

if sys.platform == "win32":
    extra_cflags += ["/std:c++20", "/Zc:preprocessor", "-DWIN32_LEAN_AND_MEAN"]
    extra_cuda_cflags += [
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
else:
    # Linux / GCC flags
    extra_cflags += ["-std=c++20", "-fPIC"]
    extra_cuda_cflags += [
        "-std=c++20",
        "--forward-unknown-opts",
        "-use_fast_math",
        "-Xcompiler",
        "-fPIC",
    ]

setup(
    name="surfel_rasterizer_extension",
    packages=['render'],
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
