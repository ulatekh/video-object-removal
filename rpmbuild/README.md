# Building gcc 4.8.5 for CUDA 6.5 under Fedora Core

* Get the source RPM [here](https://vault.centos.org/7.7.1908/os/Source/SPackages/gcc-4.8.5-39.el7.src.rpm).  (It would have been more correct to use [this one](https://dl.fedoraproject.org/pub/archive/fedora/linux/releases/20/Everything/source/SRPMS/g/gcc-4.8.2-1.fc20.src.rpm), but I was unable to get it to build.)
* Once you install the source RPM on your computer, copy the contents of the `SPECS` and `SOURCES` file to your rpmbuild directory.
* If you like, compare the installed `gcc.spec` to the copied `gcc485.spec` in this repo, to verify they're the same.
* If you like, compare `gcc485.spec` to `compat-gcc-485.spec`, to see what was involved with creating this package. (The changes revolved around getting an older version of gcc to build in Fedora Core 30, and the patches necessary for that to happen, as well as renaming the packages, executables, and so on.)
* Build the package, e.g. `nice -20 rpmbuild -ba SPECS/compat-gcc-485.spec`. On my computer, with an 8-core 3.4GHz Intel Xeon and an SSD drive, it takes about 1h45m.
* Once you've installed the packages you want (I use `compat-gcc-48`, `compat-gcc-48-c++`, `compat-libstdc++-48`, `compat-libstdc++-48-devel`, and `compat-libstdc++-48-static`), try building some of the CUDA sample files! Make sure they're using `gcc48`, `g++48`, etc.

Don't expect much support for building this; I started with the basic gcc 4.8.5 package, and Googled madly every time I ran into a problem. You can do that too. Also don't expect me to answer basic questions about building RPM packages or programming CUDA; there are plenty of better sources for that sort of info.
