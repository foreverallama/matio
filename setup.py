import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        "matio.v5._streams",
        ["matio/v5/_streams.pyx"],
        include_dirs=[numpy.get_include()],
    ),
    Extension(
        "matio.v5._mio5_utils",
        ["matio/v5/_mio5_utils.pyx"],
        include_dirs=[numpy.get_include(), "src/v5/include"],
    ),
]

setup(
    ext_modules=cythonize(extensions, include_path=["matio"]),
)
