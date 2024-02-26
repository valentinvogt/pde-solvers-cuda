# pde-solvers-cuda


## Requirements
Currently, only UNIX based operating systems are supported. Moreover, you need to have the following installed on your machine:
* C++ compiler (e.g. gcc, clang)
* CMake (at least VERSION 3.18)
* CUDA (not strictly needed)
* git (not strictly needed, you could also download the repo as .zip file)

## Getting startet
You can build pds-solvers-cuda as follows

```
git clone https://github.com/LouisHurschler/pde-solvers-cuda.git  
cd pde-solvers-cuda
mkdir build
cmkae -B build -DENABLE_CUDA={0,1}
```
Note that ENABLE_CUDA is set OFF by default
