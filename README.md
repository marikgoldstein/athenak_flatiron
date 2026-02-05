# BUILD

# step 1: clone repo

git clone --recursive https://github.com/IAS-Astrophysics/athenak.git 

the recursive is important since it also clones kokkos and athenak expects kokkos to be inside the athenak directory

# step 2: build 

make a build dir, e.g. build_athenak, and cp the build.sh from this repo in that dir, and then run it

# step 3: test

use the launch.sbatch here with the sample mri2d input file

