#!/bin/bash
src_dir=`pwd`

# build FTK for evaluation
mkdir -p ${src_dir}/external
cd ${src_dir}/external
git clone https://github.com/lxAltria/ftk.git
cd ftk
git reset --hard cf8b2bb89a4260ab4b50c38e58bb369408daf00e
cp ${src_dir}/CMakeLists_ftk.txt CMakeLists.txt
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=${src_dir}/external/ftk/install -DCMAKE_CXX_COMPILER=mpic++ -DCMAKE_C_COMPILER=mpicc
make -j 8
make install

# build cpSZ
cd ${src_dir}
mkdir build
cd build
cmake .. -DCMAKE_CXX_COMPILER=mpic++ -DCMAKE_C_COMPILER=mpicc
make -j 4
