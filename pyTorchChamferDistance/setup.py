from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cd',
    ext_modules=[
        CUDAExtension('cd_cuda', [
            'chamfer_distance.cpp',
            'chamfer_dist.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })