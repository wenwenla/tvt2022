from setuptools import Extension, setup, find_packages

RUN_EXT = Extension(
    name='cppenv._env',
    sources=[
        'cppenv/env.cpp',
        'cppenv/env.i'
    ],
    swig_opts=['-c++', '-py3'],
    extra_compile_args=['-fopenmp'],
    language='c++',
)

setup(
    name='cppenv',
    version=0.1,
    packages=find_packages(),
    ext_modules=[RUN_EXT],
)
