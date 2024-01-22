
# Preserving Topological Feature with Sign-of-Determinant Predicates in Lossy Compression: A Case Study of Vector Field Critical Points
#### This work has been accepted by ICDE2024

## Dependencies
- cmake >= 3.19.4
- gcc >= 9.3
- zstd >= 1.5.0
- ftk >= 0.0.6

## Installation
```bash
git clone https://github.com/xty9501/cpsz-sos.git
cd cpsz-sos
sh build_script.sh
cd build/bin/
```

## Dataset
In our work, we utilized four datasets: ocean (2d), hurricane, nek5000, and Turbulence. 

Datasets available online:
- [Ocean](https://github.com/szcompressor/cpSZ/tree/main/data) (uf.dat and vf.dat)
- [Hurricane](https://sdrbench.github.io/) (Uf48.bin.f32,Vf48.bin.f32,Wf48.bin.f32)
- [Nek5000](https://drive.google.com/drive/folders/1JDYp4mLebE0s0EZ2UFWJYBtxdq5km7Rz?usp=sharing) (U.dat,V.dat,W.dat)
- [Turbulence](https://turbulence.pha.jhu.edu/)

For small-scale parallel evaluation, we provide three different parti- tioning methods for the nek5000 dataset, which are: original data with dimensions 512x512x512, partitioned data with dimensions 256 x256x256x8, and partitioned data with dimensions 128x128x128x64. Please select the appropriate dataset based on the number of proces- sors used during parallel computing
## Execution
To generate the result of **table 2**,the usage of the command is as follows:


Here is an example(execution time: ∼ 30s)

```bash
# usage: cpsz_parallel <DH> <DL> <DW> <u_data_path> <v_data_path> <w_data_path> <eb> <option>*
mpirun -np 8 ./cpsz_parallel 256 256 256 ./256x256x256x8/utest ./256x256x256x8/vtest ./256x256x256x8/wtest 0.01 0
mpirun -np 8 ./cpsz_parallel 256 256 256 ./256x256x256x8/utest ./256x256x256x8/vtest ./256x256x256x8/wtest 0.01 1
mpirun -np 8 ./cpsz_parallel 256 256 256 ./256x256x256x8/utest ./256x256x256x8/vtest ./256x256x256x8/wtest 0.01 2
mpirun -np 8 ./cpsz_parallel 256 256 256 ./256x256x256x8/utest ./256x256x256x8/vtest ./256x256x256x8/wtest 0.01 3
mpirun -np 64 ./cpsz_parallel 128 128 128 ./128x128x128x64/utest ./128x128x128x64/vtest ./128x128x128x64/wtest 0.01 0
mpirun -np 64 ./cpsz_parallel 128 128 128 ./128x128x128x64/utest ./128x128x128x64/vtest ./128x128x128x64/wtest 0.01 1
mpirun -np 64 ./cpsz_parallel 128 128 128 ./128x128x128x64/utest ./128x128x128x64/vtest ./128x128x128x64/wtest 0.01 2
mpirun -np 64 ./cpsz_parallel 128 128 128 ./128x128x128x64/utest ./128x128x128x64/vtest ./128x128x128x64/wtest 0.01 3

```
The above commands will generate the compressed data(.cpsz) as well as the decompressed data(.out). At the same time, it will also print out the elapsed time of each stage, the compression ratio, and information about critical points

To generate the result for **table 3**, use the following commands:

```bash
# usage: cpsz_parallel <DH> <DL> <DW> <u_data_path> <v_data_path> <w_data_path> <eb> <option>*
mpirun -np 8 ./cpsz_parallel 256 256 256 ./256x256x256x8/utest ./256x256x256x8/vtest ./256x256x256x8/wtest 0.01 4
mpirun -np 64 ./cpsz_parallel 128 128 128 ./128x128x128x64/utest ./128x128x128x64/vtest ./128x128x128x64/wtest 0.01 4
```

To generate the result of our method for **Table 5**, use the following commands

```bash
./sz_compress_cp_preserve_2d_test <u_data> <v_data> <DL> <DW> <eb> <option>
./cp_extraction_2d_sos <u_data> <v_data> <DW> <DL>
```

Exmple:
Settings = NoSpec- R 0.1:(option = 0, eb = 0.1)
Firstly, run:
```bash
./sz_compress_cp_preserve_2d_test ./ocean/uf.dat .ocean/vf.dat 2400 3600 0.1 0
```
Then, run:
```bash
./cp_extraction_2d_sos ./ocean/uf.dat .ocean/vf.dat 3600 2400
```
For ST1, ST2, ST3 and ST4, simply just change <option> to 1, 2, 3, 4

To generate the result for **Table 6**, use the following commands:
```bash
./sz_compress_cp_preserve_3d_test <u_data> <v_data> <w_data> <DL> <DW> <DH> <eb> <option>
./cp_extraction_3d_sos <u_data> <v_data> <w_data> <DH> <DW> <DL>
```

Similarly, to generate the result for Table 7, change <u_data> and <v_data> corresponds to the path of nek5000 dataset. 

We compared our method with various other compressors. For their results, please refer to:
- [FPZIP](https://github.com/LLNL/fpzip)
- [ZFP](https://github.com/LLNL/zfp)
- [SZ3](https://github.com/szcompressor/SZ3)
- [cpSZ](https://github.com/szcompressor/cpSZ)

## Contributions
1. We develop a general theory to derive the allowable perturbation for one
row of a matrix while preserving its sign of determinant. We then apply
this theory to preserve critical points in vector fields, because critical point
detection can be reduced to the result of point-in-simplex test that purely
relies on the sign of determinants
2. We also optimize this algorithm with a speculative compression scheme to
allow for high compression ratios and efficiently parallelize it in distributed
environments.
3. We perform solid experiments with real-world datasets, demonstrating
that our method leads to up to 440% improvements on compression ratios
over state-of-the-art lossy compressors when all the critical points need to
be preserved. Using the parallelization strategies, our method delivers 2×
performance speedup in data loading compared with reading data without
compression.
