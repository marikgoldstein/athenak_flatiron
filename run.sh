module purge

# old docs
#module load modules/2.1-20230203 slurm cuda/11.8.0 openmpi/cuda-4.0.7

# new instructions
module load gcc cuda openmpi

export LD_PRELOAD=/mnt/sw/fi/cephtweaks/lib/libcephtweaks.so
export CEPHTWEAKS_LAZYIO=1

WHERE="/mnt/home/mgoldstein/athenak"
H_FLAGS="-D CMAKE_CXX_COMPILER=g++ -D CMAKE_BUILD_TYPE=Release  -D Athena_ENABLE_MPI=ON -D Kokkos_ENABLE_CUDA=ON -D Kokkos_ARCH_HOPPER90=ON -D Kokkos_ARCH_NATIVE=ON"
A_FLAGS="-D CMAKE_CXX_COMPILER=${WHERE}/kokkos/bin/nvcc_wrapper -D Athena_ENABLE_MPI=On -D Kokkos_ENABLE_CUDA=On -D Kokkos_ARCH_AMPERE80=On -D Kokkos_ARCH_NATIVE=ON"

cmake ${WHERE} ${A_FLAGS}
cmake --build . -j 30
