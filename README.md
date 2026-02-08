
# step 1: clone repo

```
git clone --recursive https://github.com/IAS-Astrophysics/athenak.git
```
the recursive is important since it also clones kokkos and athenak expects kokkos to be inside the athenak directory


# step 2 checkout:

go to athenak subdir and do 

```
git checkout 92bbc3f21d7768232d4aaf1ecbc49c46627e891b
```

# step 3: make changes

really just change
basically, just change 

```
pmbp->pmhd->psrc
``` 
to 
```
pmbp->pmhd->psbox_u
``` 
in 
```
athenak/src/pgen/mri2d.cpp
```

in more detail, the diff is:

```
diff --git a/src/pgen/mri2d.cpp b/src/pgen/mri2d.cpp
index 3947cdee..627e9b95 100644
--- a/src/pgen/mri2d.cpp
+++ b/src/pgen/mri2d.cpp
@@ -49,7 +49,7 @@ void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
   }
   MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
   if (pmbp->pmhd != nullptr) {
-    if (pmbp->pmhd->psrc == nullptr) {
+    if (pmbp->pmhd->psbox_u == nullptr) {
       std::cout <<"### FATAL ERROR in "<< __FILE__ <<" at line " <<__LINE__ << std::endl
                 << "Shearing box source terms not enabled for mri2d problem" << std::endl;
       Exit(EXIT_FAILURE);
```

# step 4: build 

# run build.sh

# step 5: test 
use the launch.sbatch here with the sample mri2d input file

