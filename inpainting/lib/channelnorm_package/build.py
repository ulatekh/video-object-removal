import os
import torch
import torch.utils.ffi
from distutils import sysconfig
import sys, re

# Remove compiler flags that don't work with gcc 4.8.5
# Technique taken from https://stackoverflow.com/a/21577015/603828
# TODO: Detect gcc version.
if sys.platform == 'linux' or sys.platform == 'darwin':
    sysconfig.get_config_var(None)  # to fill up _config_vars
    d = sysconfig._config_vars
    for x in ['OPT', 'CFLAGS', 'PY_CFLAGS', 'PY_CORE_CFLAGS', 'CONFIGURE_CFLAGS', 'LDSHARED']:
        d[x] = re.sub(' -fcf-protection ', ' ', d[x])
        d[x] = re.sub('^-fcf-protection ', '',  d[x])
        d[x] = re.sub(' -fcf-protection$', '',  d[x])

this_folder = os.path.dirname(os.path.abspath(__file__)) + '/'

Headers = []
Sources = []
Defines = []
Objects = []

if torch.cuda.is_available() == True:
    Headers += ['src/ChannelNorm_cuda.h']
    Sources += ['src/ChannelNorm_cuda.c']
    Defines += [('WITH_CUDA', None)]
    Objects += ['src/ChannelNorm_kernel.o']

ffi = torch.utils.ffi.create_extension(
    name='_ext.channelnorm',
    headers=Headers,
    sources=Sources,
    verbose=False,
    with_cuda=True,
    package=False,
    relative_to=this_folder,
    define_macros=Defines,
    extra_objects=[os.path.join(this_folder, Object) for Object in Objects],
    extra_compile_args=["-std=c++11"]
)

if __name__ == '__main__':
    ffi.build()
