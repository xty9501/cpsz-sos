
#include <stdio.h>
#include <mutex>
#include <ftk/numeric/print.hh>
#include <ftk/numeric/cross_product.hh>
#include <ftk/numeric/vector_norm.hh>
#include <ftk/numeric/linear_interpolation.hh>
#include <ftk/numeric/bilinear_interpolation.hh>
#include <ftk/numeric/inverse_linear_interpolation_solver.hh>
#include <ftk/numeric/inverse_bilinear_interpolation_solver.hh>
#include <ftk/numeric/gradient.hh>
#include <ftk/algorithms/cca.hh>
#include <ftk/geometry/cc2curves.hh>
#include <ftk/geometry/curve2tube.hh>
#include <ftk/ndarray.hh>
#include <ftk/mesh/simplicial_regular_mesh.hh>
#include <ftk/numeric/critical_point_type.hh>
#include <ftk/numeric/critical_point_test.hh>
#include <unordered_map>
#include <vector>
#include <queue>
#include <fstream>
#include <ctime>
#include <mpi.h>
#include "sz_lossless.hpp"
#include "io_utils.hpp"

#include "sz_cp_preserve_utils.hpp"
#include "sz_def.hpp"
#include "sz_compression_utils.hpp"
#include <ftk/numeric/fixed_point.hh>
#include "sz_compress_cp_preserve_sos_3d_parallel.hpp"
#include "sz_decompress_cp_preserve_3d_parallel.hpp"
#include "sz_decompress_cp_preserve_3d.hpp"

struct critical_point_t
{
  double x[3];
  int type;
  size_t simplex_id;
  critical_point_t() {}
};

#define SINGULAR 0
#define STABLE_SOURCE 1
#define UNSTABLE_SOURCE 2
#define STABLE_REPELLING_SADDLE 3
#define UNSTABLE_REPELLING_SADDLE 4
#define STABLE_ATRACTTING_SADDLE 5
#define UNSTABLE_ATRACTTING_SADDLE 6
#define STABLE_SINK 7
#define UNSTABLE_SINK 8

static const int tet_coords[6][4][3] = {
    {{0, 0, 0},
     {0, 0, 1},
     {0, 1, 1},
     {1, 1, 1}},
    {{0, 0, 0},
     {0, 1, 0},
     {0, 1, 1},
     {1, 1, 1}},
    {{0, 0, 0},
     {0, 0, 1},
     {1, 0, 1},
     {1, 1, 1}},
    {{0, 0, 0},
     {1, 0, 0},
     {1, 0, 1},
     {1, 1, 1}},
    {{0, 0, 0},
     {0, 1, 0},
     {1, 1, 0},
     {1, 1, 1}},
    {{0, 0, 0},
     {1, 0, 0},
     {1, 1, 0},
     {1, 1, 1}},
};

template <typename T_acc, typename T>
static inline void
update_value(T_acc v[4][3], int local_id, int global_id, const T *U, const T *V, const T *W)
{
  v[local_id][0] = U[global_id];
  v[local_id][1] = V[global_id];
  v[local_id][2] = W[global_id];
}

template <typename T, typename T_fp>
static inline void
update_index_and_value(double v[4][3], int64_t vf[4][3], int indices[4], int local_id, int global_id, const T *U, const T *V, const T *W, const T_fp *U_fp, const T_fp *V_fp, const T_fp *W_fp)
{
  indices[local_id] = global_id;
  update_value(v, local_id, global_id, U, V, W);
  update_value(vf, local_id, global_id, U_fp, V_fp, W_fp);
}

template <typename T_fp>
static void
check_simplex_seq(const T_fp vf[4][3], const double v[4][3], const double X[3][3], const int indices[4], int i, int j, int k, int simplex_id, std::unordered_map<int, critical_point_t> &critical_points)
{
  // robust critical point test
  bool succ = ftk::robust_critical_point_in_simplex3(vf, indices);
  if (!succ)
    return;
  double mu[4]; // check intersection
  double cond;
  bool succ2 = ftk::inverse_lerp_s3v3(v, mu, &cond);
  if (!succ2)
    ftk::clamp_barycentric<4>(mu);
  double x[3]; // position
  ftk::lerp_s3v3(X, mu, x);
  critical_point_t cp;
  cp.x[0] = k + x[0];
  cp.x[1] = j + x[1];
  cp.x[2] = i + x[2];
  cp.type = get_cp_type(X, v);
  cp.simplex_id = simplex_id;
  critical_points[simplex_id] = cp;
}

template <typename T>
std::unordered_map<int, critical_point_t>
compute_critical_points(const T *U, const T *V, const T *W, int r1, int r2, int r3, uint64_t vector_field_scaling_factor)
{
  // check cp for all cells
  using T_fp = int64_t;
  ptrdiff_t dim0_offset = r2 * r3;
  ptrdiff_t dim1_offset = r3;
  ptrdiff_t cell_dim0_offset = (r2 - 1) * (r3 - 1);
  ptrdiff_t cell_dim1_offset = r3 - 1;
  size_t num_elements = r1 * r2 * r3;
  T_fp *U_fp = (T_fp *)malloc(num_elements * sizeof(T_fp));
  T_fp *V_fp = (T_fp *)malloc(num_elements * sizeof(T_fp));
  T_fp *W_fp = (T_fp *)malloc(num_elements * sizeof(T_fp));
  for (int i = 0; i < num_elements; i++)
  {
    U_fp[i] = U[i] * vector_field_scaling_factor;
    V_fp[i] = V[i] * vector_field_scaling_factor;
    W_fp[i] = W[i] * vector_field_scaling_factor;
  }
  int indices[4] = {0};
  // __int128 vf[4][3] = {0};
  int64_t vf[4][3] = {0};
  double v[4][3] = {0};
  double actual_coords[6][4][3];
  for (int i = 0; i < 6; i++)
  {
    for (int j = 0; j < 4; j++)
    {
      for (int k = 0; k < 3; k++)
      {
        actual_coords[i][j][k] = tet_coords[i][j][k];
      }
    }
  }
  std::unordered_map<int, critical_point_t> critical_points;
  for (int i = 0; i < r1 - 1; i++)
  {
    // if (i % 200 == 0)
    //   std::cout << i << " / " << r1 - 1 << std::endl;
    for (int j = 0; j < r2 - 1; j++)
    {
      for (int k = 0; k < r3 - 1; k++)
      {
        // order (reserved, z->x):
        // ptrdiff_t cell_offset = 6*(i*cell_dim0_offset + j*cell_dim1_offset + k);
        // ftk index
        ptrdiff_t cell_offset = 6 * (i * dim0_offset + j * dim1_offset + k);
        // (ftk-0) 000, 001, 011, 111
        update_index_and_value(v, vf, indices, 0, i * dim0_offset + j * dim1_offset + k, U, V, W, U_fp, V_fp, W_fp);
        update_index_and_value(v, vf, indices, 1, (i + 1) * dim0_offset + j * dim1_offset + k, U, V, W, U_fp, V_fp, W_fp);
        update_index_and_value(v, vf, indices, 2, (i + 1) * dim0_offset + (j + 1) * dim1_offset + k, U, V, W, U_fp, V_fp, W_fp);
        update_index_and_value(v, vf, indices, 3, (i + 1) * dim0_offset + (j + 1) * dim1_offset + (k + 1), U, V, W, U_fp, V_fp, W_fp);
        check_simplex_seq(vf, v, actual_coords[0], indices, i, j, k, cell_offset, critical_points);
        // (ftk-2) 000, 010, 011, 111
        update_index_and_value(v, vf, indices, 1, i * dim0_offset + (j + 1) * dim1_offset + k, U, V, W, U_fp, V_fp, W_fp);
        check_simplex_seq(vf, v, actual_coords[1], indices, i, j, k, cell_offset + 2, critical_points);
        // (ftk-1) 000, 001, 101, 111
        update_index_and_value(v, vf, indices, 1, (i + 1) * dim0_offset + j * dim1_offset + k, U, V, W, U_fp, V_fp, W_fp);
        update_index_and_value(v, vf, indices, 2, (i + 1) * dim0_offset + j * dim1_offset + k + 1, U, V, W, U_fp, V_fp, W_fp);
        check_simplex_seq(vf, v, actual_coords[2], indices, i, j, k, cell_offset + 1, critical_points);
        // (ftk-4) 000, 100, 101, 111
        update_index_and_value(v, vf, indices, 1, i * dim0_offset + j * dim1_offset + k + 1, U, V, W, U_fp, V_fp, W_fp);
        check_simplex_seq(vf, v, actual_coords[3], indices, i, j, k, cell_offset + 4, critical_points);
        // (ftk-3) 000, 010, 110, 111
        update_index_and_value(v, vf, indices, 1, i * dim0_offset + (j + 1) * dim1_offset + k, U, V, W, U_fp, V_fp, W_fp);
        update_index_and_value(v, vf, indices, 2, i * dim0_offset + (j + 1) * dim1_offset + k + 1, U, V, W, U_fp, V_fp, W_fp);
        check_simplex_seq(vf, v, actual_coords[4], indices, i, j, k, cell_offset + 3, critical_points);
        // (ftk-5) 000, 100, 110, 111
        update_index_and_value(v, vf, indices, 1, i * dim0_offset + j * dim1_offset + k + 1, U, V, W, U_fp, V_fp, W_fp);
        check_simplex_seq(vf, v, actual_coords[5], indices, i, j, k, cell_offset + 5, critical_points);
      }
    }
  }
  free(U_fp);
  free(V_fp);
  free(W_fp);
  return critical_points;
}

bool compare_float_arrays(float *arr1, float *arr2, int length)
{
  for (int i = 0; i < length; i++)
  {
    if (arr1[i] != arr2[i])
    {
      return false;
    }
  }
  return true;
}

bool check_if_original_data_assigned(float *ori_data, float *new_data, int new_xn, int new_yn, int new_zn)
{
  float *ori_data_start = ori_data;
  float *new_data_start = new_data;
  for (size_t z = 0; z < 128; z++)
  {
    for (size_t y = 0; y < 128; y++)
    {
      if (!compare_float_arrays(ori_data, new_data + z * (new_xn * new_yn) + (y * new_yn), 128))
        return false;
      ori_data += 128;
    }
  }
  ori_data = ori_data_start;
  new_data = new_data_start;
  return true;
}

int file_size(char* filename)
{
    struct stat statbuf;
    stat(filename,&statbuf);
    size_t size=statbuf.st_size;
    return size/sizeof(float);
}

void cover_ghost_points(float *u, float *v, float *w, float *new_U, float *new_V, float *new_W,
                        int &DL, int &DH, int &DW,int &xn, int &yn, int &zn, int &front, int &back, int &top, int &bot, int &left, int &right,
                        int *coords, MPI_Comm comm_cart,MPI_Comm mpi_comm, MPI_Status status)
{
  float *u_pos = u;
  float *v_pos = v;
  float *w_pos = w;
  for (size_t z = 0; z < DL; z++)
  {
    for (size_t y = 0; y < DH; y++)
    {
      memcpy(new_U + z * (xn * yn) + y * xn, u_pos, DW * sizeof(float));
      memcpy(new_V + z * (xn * yn) + y * xn, v_pos, DW * sizeof(float));
      memcpy(new_W + z * (xn * yn) + y * xn, w_pos, DW * sizeof(float));
      u_pos += DW;
      v_pos += DW;
      w_pos += DW;
    }
  }
  // MPI_Barrier(MPI_COMM_WORLD);

  // send bot surface
  float *bot_surface_u = (float *)malloc(sizeof(float) * DW * DH); // N*N,
  float *bot_surface_v = (float *)malloc(sizeof(float) * DW * DH); // N*N
  float *bot_surface_w = (float *)malloc(sizeof(float) * DW * DH); // N*N
  if (bot != 1)
  {
    memcpy(bot_surface_u, u, sizeof(float) * DW * DH);
    memcpy(bot_surface_v, v, sizeof(float) * DW * DH);
    memcpy(bot_surface_w, w, sizeof(float) * DW * DH);
    int bot_coords[3] = {coords[0] - 1, coords[1], coords[2]}; // get bot block's coord
    int bot_rank;
    MPI_Cart_rank(comm_cart, bot_coords, &bot_rank);
    MPI_Send(bot_surface_u, DW * DH, MPI_FLOAT, bot_rank, 0, MPI_COMM_WORLD);
    MPI_Send(bot_surface_v, DW * DH, MPI_FLOAT, bot_rank, 1, MPI_COMM_WORLD);
    MPI_Send(bot_surface_w, DW * DH, MPI_FLOAT, bot_rank, 2, MPI_COMM_WORLD);
    free(bot_surface_u);
    free(bot_surface_v);
    free(bot_surface_w);
  }
  // recv & copy to top surface
  float *recvied_from_top_u = (float *)malloc(sizeof(float) * DW * DH);
  float *recvied_from_top_v = (float *)malloc(sizeof(float) * DW * DH);
  float *recvied_from_top_w = (float *)malloc(sizeof(float) * DW * DH);
  float *recvied_from_top_u_pos = recvied_from_top_u;
  float *recvied_from_top_v_pos = recvied_from_top_v;
  float *recvied_from_top_w_pos = recvied_from_top_w;
  if (top != 1)
  {
    // recv data from top block
    int top_coords[3] = {coords[0] + 1, coords[1], coords[2]}; // get top block's coord
    int top_rank;
    MPI_Cart_rank(comm_cart, top_coords, &top_rank);
    MPI_Recv(recvied_from_top_u_pos, DW * DH, MPI_FLOAT, top_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(recvied_from_top_v_pos, DW * DH, MPI_FLOAT, top_rank, 1, MPI_COMM_WORLD, &status);
    MPI_Recv(recvied_from_top_w_pos, DW * DH, MPI_FLOAT, top_rank, 2, MPI_COMM_WORLD, &status);
    for (size_t i = 0; i < DH; i++)
    {
      // printf("i = %d",i);
      memcpy(new_U + xn * yn * (zn - 1) + i * xn, recvied_from_top_u_pos, sizeof(float) * DW);
      memcpy(new_V + xn * yn * (zn - 1) + i * xn, recvied_from_top_v_pos, sizeof(float) * DW);
      memcpy(new_W + xn * yn * (zn - 1) + i * xn, recvied_from_top_w_pos, sizeof(float) * DW);
      recvied_from_top_u_pos += DW;
      recvied_from_top_v_pos += DW;
      recvied_from_top_w_pos += DW;
    }
    free(recvied_from_top_u);
    free(recvied_from_top_v);
    free(recvied_from_top_w);
  }

  //////////////////////////
  // send left surface
  if (left != 1)
  {
    // send left surface to left block
    float *left_surface_u = (float *)malloc(sizeof(float) * DH * DL); // N*N
    float *left_surface_v = (float *)malloc(sizeof(float) * DH * DL); // N*N
    float *left_surface_w = (float *)malloc(sizeof(float) * DH * DL); // N*N
    float *left_surface_u_pos = left_surface_u;
    float *left_surface_v_pos = left_surface_v;
    float *left_surface_w_pos = left_surface_w;
    for (size_t i = 0; i < DW * DH * DL; i += DW)
    {
      *left_surface_u_pos = u[i];
      left_surface_u_pos++;
      *left_surface_v_pos = v[i];
      left_surface_v_pos++;
      *left_surface_w_pos = w[i];
      left_surface_w_pos++;
    }
    int left_coords[3] = {coords[0], coords[1], coords[2] - 1}; // get left block's coord
    int left_rank;
    MPI_Cart_rank(comm_cart, left_coords, &left_rank);
    MPI_Send(left_surface_u, DH * DL, MPI_FLOAT, left_rank, 3, MPI_COMM_WORLD);
    MPI_Send(left_surface_v, DH * DL, MPI_FLOAT, left_rank, 4, MPI_COMM_WORLD);
    MPI_Send(left_surface_w, DH * DL, MPI_FLOAT, left_rank, 5, MPI_COMM_WORLD);

    free(left_surface_u);
    free(left_surface_v);
    free(left_surface_w);
  }
  if (right != 1)
  {
    // recv data from right block
    int right_coords[3] = {coords[0], coords[1], coords[2] + 1}; // get right block's coord
    int right_rank;
    float *recvied_from_right_u = (float *)malloc(sizeof(float) * DH * DL);
    float *recvied_from_right_v = (float *)malloc(sizeof(float) * DH * DL);
    float *recvied_from_right_w = (float *)malloc(sizeof(float) * DH * DL);
    MPI_Cart_rank(comm_cart, right_coords, &right_rank);
    MPI_Recv(recvied_from_right_u, DH * DL, MPI_FLOAT, right_rank, 3, MPI_COMM_WORLD, &status);
    MPI_Recv(recvied_from_right_v, DH * DL, MPI_FLOAT, right_rank, 4, MPI_COMM_WORLD, &status);
    MPI_Recv(recvied_from_right_w, DH * DL, MPI_FLOAT, right_rank, 5, MPI_COMM_WORLD, &status);

    float *recvied_from_right_u_pos = recvied_from_right_u;
    float *recvied_from_right_v_pos = recvied_from_right_v;
    float *recvied_from_right_w_pos = recvied_from_right_w;
    for (size_t z = 0; z < DL; z++)
    {
      for (size_t y = 0; y < DH; y++)
      {
        new_U[(xn - 1) + y * xn + z * xn * yn] = *recvied_from_right_u_pos++;
        new_V[(xn - 1) + y * xn + z * xn * yn] = *recvied_from_right_v_pos++;
        new_W[(xn - 1) + y * xn + z * xn * yn] = *recvied_from_right_w_pos++;
      }
    }
    free(recvied_from_right_u);
    free(recvied_from_right_v);
    free(recvied_from_right_w);
  }
  //////////////////////////
  // send back surface
  if (back != 1)
  {
    // send back surface to back block
    float *back_surface_u = (float *)malloc(sizeof(float) * DL * DW); // N*N
    float *back_surface_v = (float *)malloc(sizeof(float) * DL * DW); // N*N
    float *back_surface_w = (float *)malloc(sizeof(float) * DL * DW); // N*N
    float *back_surface_u_pos = back_surface_u;
    float *back_surface_v_pos = back_surface_v;
    float *back_surface_w_pos = back_surface_w;
    for (size_t i = 0; i < DW * DH * DL; i += DW * DH)
    {
      memcpy(back_surface_u_pos, u + i, sizeof(float) * DW);
      back_surface_u_pos += DW;
      memcpy(back_surface_v_pos, v + i, sizeof(float) * DW);
      back_surface_v_pos += DW;
      memcpy(back_surface_w_pos, w + i, sizeof(float) * DW);
      back_surface_w_pos += DW;
    }
    int back_coords[3] = {coords[0], coords[1] - 1, coords[2]}; // get back block's coord
    int back_rank;
    MPI_Cart_rank(comm_cart, back_coords, &back_rank);
    MPI_Send(back_surface_u, DL * DW, MPI_FLOAT, back_rank, 6, MPI_COMM_WORLD);
    MPI_Send(back_surface_v, DL * DW, MPI_FLOAT, back_rank, 7, MPI_COMM_WORLD);
    MPI_Send(back_surface_w, DL * DW, MPI_FLOAT, back_rank, 8, MPI_COMM_WORLD);
    free(back_surface_u);
    free(back_surface_v);
    free(back_surface_w);
  }
  if (front != 1)
  {
    // recv data from front block
    float *recvied_from_front_u = (float *)malloc(sizeof(float) * DL * DW);
    float *recvied_from_front_v = (float *)malloc(sizeof(float) * DL * DW);
    float *recvied_from_front_w = (float *)malloc(sizeof(float) * DL * DW);

    int front_coords[3] = {coords[0], coords[1] + 1, coords[2]}; // get front block's coord
    int front_rank;
    MPI_Cart_rank(comm_cart, front_coords, &front_rank);
    MPI_Recv(recvied_from_front_u, DL * DW, MPI_FLOAT, front_rank, 6, MPI_COMM_WORLD, &status);
    MPI_Recv(recvied_from_front_v, DL * DW, MPI_FLOAT, front_rank, 7, MPI_COMM_WORLD, &status);
    MPI_Recv(recvied_from_front_w, DL * DW, MPI_FLOAT, front_rank, 8, MPI_COMM_WORLD, &status);

    float *recvied_from_front_u_pos = recvied_from_front_u;
    float *recvied_from_front_v_pos = recvied_from_front_v;
    float *recvied_from_front_w_pos = recvied_from_front_w;
    for (size_t z = 0; z < DL; z++)
    {
      size_t offset = xn * yn - xn;
      memcpy(new_U + z * xn * yn + offset, recvied_from_front_u_pos, DW * sizeof(float));
      memcpy(new_V + z * xn * yn + offset, recvied_from_front_v_pos, DW * sizeof(float));
      memcpy(new_W + z * xn * yn + offset, recvied_from_front_w_pos, DW * sizeof(float));
      recvied_from_front_u_pos += DW;
      recvied_from_front_v_pos += DW;
      recvied_from_front_w_pos += DW;
    }

    free(recvied_from_front_u);
    free(recvied_from_front_v);
    free(recvied_from_front_w);
  }

  ///////////
  // send & recv 3 edges

  // edge parallel to x axis, recv from front top block
  if (bot != 1 && back != 1)
  { 
    //if bot back block exists, send edge to bot back block
    int bot_back_coords[3] = {coords[0] - 1, coords[1] - 1, coords[2]};
    int bot_back_rank;
    MPI_Cart_rank(comm_cart, bot_back_coords, &bot_back_rank);
    MPI_Send(u, DW, MPI_FLOAT, bot_back_rank, 9, MPI_COMM_WORLD);
    MPI_Send(v, DW, MPI_FLOAT, bot_back_rank, 10, MPI_COMM_WORLD);
    MPI_Send(w, DW, MPI_FLOAT, bot_back_rank, 11, MPI_COMM_WORLD);
  }
  if (front != 1 && top != 1)
  { 
    //if front top block exists, recv edge from front top block
    int front_top_coords[3] = {coords[0] + 1, coords[1] + 1, coords[2]};
    int front_top_rank;
    MPI_Cart_rank(comm_cart, front_top_coords, &front_top_rank);
    MPI_Recv(new_U + xn * yn * zn - xn, DW, MPI_FLOAT, front_top_rank, 9, MPI_COMM_WORLD, &status);
    MPI_Recv(new_V + xn * yn * zn - xn, DW, MPI_FLOAT, front_top_rank, 10, MPI_COMM_WORLD, &status);
    MPI_Recv(new_W + xn * yn * zn - xn, DW, MPI_FLOAT, front_top_rank, 11, MPI_COMM_WORLD, &status);
  }
  //////////


  // edge parallel to y axis, recv from right top block
  if (left != 1 && bot != 1)
  { 
    //if left bot block exists, send edge to left bot block
    int left_bot_coords[3] = {coords[0] - 1, coords[1], coords[2] - 1};
    int left_bot_rank;
    MPI_Cart_rank(comm_cart, left_bot_coords, &left_bot_rank);
    float *y_axis_buffer_u = (float *)malloc(sizeof(float) * DH);
    float *y_axis_buffer_v = (float *)malloc(sizeof(float) * DH);
    float *y_axis_buffer_w = (float *)malloc(sizeof(float) * DH);
    float *y_axis_buffer_u_pos = y_axis_buffer_u;
    float *y_axis_buffer_v_pos = y_axis_buffer_v;
    float *y_axis_buffer_w_pos = y_axis_buffer_w;
    for (size_t i = 0; i < DW * DH; i += DW)
    {
      *y_axis_buffer_u_pos = u[i];
      y_axis_buffer_u_pos++;
      *y_axis_buffer_v_pos = v[i];
      y_axis_buffer_v_pos++;
      *y_axis_buffer_w_pos = w[i];
      y_axis_buffer_w_pos++;
    }
    MPI_Send(y_axis_buffer_u, DH, MPI_FLOAT, left_bot_rank, 12, MPI_COMM_WORLD);
    MPI_Send(y_axis_buffer_v, DH, MPI_FLOAT, left_bot_rank, 13, MPI_COMM_WORLD);
    MPI_Send(y_axis_buffer_w, DH, MPI_FLOAT, left_bot_rank, 14, MPI_COMM_WORLD);
    free(y_axis_buffer_u);
    free(y_axis_buffer_v);
    free(y_axis_buffer_w);
  }

  if (top != 1 && right != 1)
  {
    int top_right_coords[3] = {coords[0] + 1, coords[1], coords[2] + 1};
    int top_right_rank;
    MPI_Cart_rank(comm_cart, top_right_coords, &top_right_rank);
    float *recv_y_buffer_u = (float *)malloc(sizeof(float) * DH);
    float *recv_y_buffer_v = (float *)malloc(sizeof(float) * DH);
    float *recv_y_buffer_w = (float *)malloc(sizeof(float) * DH);

    MPI_Recv(recv_y_buffer_u, DH, MPI_FLOAT, top_right_rank, 12, MPI_COMM_WORLD, &status);
    MPI_Recv(recv_y_buffer_v, DH, MPI_FLOAT, top_right_rank, 13, MPI_COMM_WORLD, &status);
    MPI_Recv(recv_y_buffer_w, DH, MPI_FLOAT, top_right_rank, 14, MPI_COMM_WORLD, &status);

    float *recv_y_buffer_u_pos = recv_y_buffer_u;
    float *recv_y_buffer_v_pos = recv_y_buffer_v;
    float *recv_y_buffer_w_pos = recv_y_buffer_w;
    for (size_t i = xn * yn * (zn - 1) + (xn - 1); i < xn * yn * zn; i += xn)
    {
      new_U[i] = *recv_y_buffer_u_pos;
      new_V[i] = *recv_y_buffer_v_pos;
      new_W[i] = *recv_y_buffer_w_pos;
      recv_y_buffer_u_pos++;
      recv_y_buffer_v_pos++;
      recv_y_buffer_w_pos++;
    }
    free(recv_y_buffer_u);
    free(recv_y_buffer_v);
    free(recv_y_buffer_w);
  }
  /////////////
  // edge parallel to z axis, recv from left front block
  if (left != 1 && back != 1)
  {
    // send data
    int left_back_coords[3] = {coords[0], coords[1] - 1, coords[2] - 1};
    int left_back_rank;
    MPI_Cart_rank(comm_cart, left_back_coords, &left_back_rank);
    float *z_axis_buffer_u = (float *)malloc(sizeof(float) * DL);
    float *z_axis_buffer_v = (float *)malloc(sizeof(float) * DL);
    float *z_axis_buffer_w = (float *)malloc(sizeof(float) * DL);
    float *z_axis_buffer_u_pos = z_axis_buffer_u;
    float *z_axis_buffer_v_pos = z_axis_buffer_v;
    float *z_axis_buffer_w_pos = z_axis_buffer_w;
    for (size_t i = 0; i < DW * DH * DL; i += DW * DH)
    {
      *z_axis_buffer_u_pos = u[i];
      z_axis_buffer_u_pos++;
      *z_axis_buffer_v_pos = v[i];
      z_axis_buffer_v_pos++;
      *z_axis_buffer_w_pos = w[i];
      z_axis_buffer_w_pos++;
    }
    MPI_Send(z_axis_buffer_u, DL, MPI_FLOAT, left_back_rank, 15, MPI_COMM_WORLD);
    MPI_Send(z_axis_buffer_v, DL, MPI_FLOAT, left_back_rank, 16, MPI_COMM_WORLD);
    MPI_Send(z_axis_buffer_w, DL, MPI_FLOAT, left_back_rank, 17, MPI_COMM_WORLD);
    free(z_axis_buffer_u);
    free(z_axis_buffer_v);
    free(z_axis_buffer_w);
  }

  if (front != 1 && right != 1)
  {
    // recv data
    int front_right_coords[3] = {coords[0], coords[1] + 1, coords[2] + 1};
    int front_right_rank;
    MPI_Cart_rank(comm_cart, front_right_coords, &front_right_rank);
    float *recv_z_buffer_u = (float *)malloc(sizeof(float) * DL);
    float *recv_z_buffer_v = (float *)malloc(sizeof(float) * DL);
    float *recv_z_buffer_w = (float *)malloc(sizeof(float) * DL);
    MPI_Recv(recv_z_buffer_u, DL, MPI_FLOAT, front_right_rank, 15, MPI_COMM_WORLD, &status);
    MPI_Recv(recv_z_buffer_v, DL, MPI_FLOAT, front_right_rank, 16, MPI_COMM_WORLD, &status);
    MPI_Recv(recv_z_buffer_w, DL, MPI_FLOAT, front_right_rank, 17, MPI_COMM_WORLD, &status);

    float *recv_z_buffer_u_pos = recv_z_buffer_u;
    float *recv_z_buffer_v_pos = recv_z_buffer_v;
    float *recv_z_buffer_w_pos = recv_z_buffer_w;
    for (size_t i = (xn * yn) - 1; i < xn * yn * zn; i += xn * yn)
    {
      new_U[i] = *recv_z_buffer_u_pos;
      recv_z_buffer_u_pos++;
      new_V[i] = *recv_z_buffer_v_pos;
      recv_z_buffer_v_pos++;
      new_W[i] = *recv_z_buffer_w_pos;
      recv_z_buffer_w_pos++;
    }
    free(recv_z_buffer_u);
    free(recv_z_buffer_v);
    free(recv_z_buffer_w);
  }

  // send single point
  if (bot != 1 && left != 1 && back != 1)
  {
    int bot_left_back_coords[3] = {coords[0] - 1, coords[1] - 1, coords[2] - 1};
    int bot_left_back_rank;
    MPI_Cart_rank(comm_cart, bot_left_back_coords, &bot_left_back_rank);
    MPI_Send(u, 1, MPI_FLOAT, bot_left_back_rank, 18, MPI_COMM_WORLD);
    MPI_Send(v, 1, MPI_FLOAT, bot_left_back_rank, 19, MPI_COMM_WORLD);
    MPI_Send(w, 1, MPI_FLOAT, bot_left_back_rank, 20, MPI_COMM_WORLD);
  }

  // recv single point
  if (front != 1 && top != 1 && right != 1)
  {
    int right_top_front_coords[3] = {coords[0] + 1, coords[1] + 1, coords[2] + 1};
    int right_top_front_rank;
    MPI_Cart_rank(comm_cart, right_top_front_coords, &right_top_front_rank);
    MPI_Recv(new_U + xn * yn * zn - 1, 1, MPI_FLOAT, right_top_front_rank, 18, MPI_COMM_WORLD, &status);
    MPI_Recv(new_V + xn * yn * zn - 1, 1, MPI_FLOAT, right_top_front_rank, 19, MPI_COMM_WORLD, &status);
    MPI_Recv(new_W + xn * yn * zn - 1, 1, MPI_FLOAT, right_top_front_rank, 20, MPI_COMM_WORLD, &status);
  }
}

int main(int argc, char **argv)
{

  int rank, size;
  // int dims[2]; //2D case
  // int periods[2] = {0,0}; // 2D case
  // int coords[2]; // 2D case
  int dims[3];
  int periods[3] = {0, 0, 0};
  int coords[3];
  MPI_Status status;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int size_cubert = std::round(std::pow(size, 1.0 / 3));
  dims[0] = dims[1] = dims[2] = size_cubert; 
  double compression_time = 0, writing_time = 0, reading_time = 0, decompression_time = 0,reading_compressed_data_time = 0,
  write_compressed_data_time = 0,buid_ghost_point_time = 0;
  // printf("size-cubert = %d\n",size_cubert);

  MPI_Comm comm_cart;
  MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &comm_cart);

  MPI_Cart_coords(comm_cart, rank, 3, coords);

  MPI_Barrier(MPI_COMM_WORLD);
  // printf("Rank %d has coordinates (%d, %d,%d)\n", rank, coords[0], coords[1],coords[2]);
  // exit(0);
  int DW = atoi(argv[1]); // row
  int DH = atoi(argv[2]); // col
  int DL = atoi(argv[3]); // height
  int sos = 1;
  size_t num = 0;
  double elapsed_time;
  std::string u_file_name = (std::string(argv[4]) + "_" + std::to_string(coords[0] + 1) +
                             "_" + std::to_string(coords[1] + 1) + "_" + std::to_string(coords[2] + 1) + ".dat");
  std::string v_file_name = (std::string(argv[5]) + "_" + std::to_string(coords[0] + 1) +
                             "_" + std::to_string(coords[1] + 1) + "_" + std::to_string(coords[2] + 1) + ".dat");
  std::string w_file_name = (std::string(argv[6]) + "_" + std::to_string(coords[0] + 1) +
                             "_" + std::to_string(coords[1] + 1) + "_" + std::to_string(coords[2] + 1) + ".dat");

  MPI_Barrier(MPI_COMM_WORLD);
	if(rank == 0) elapsed_time = -MPI_Wtime();
  float * u = (float *) malloc(DW*DH*DL * sizeof(float));
  float * v = (float *) malloc(DW*DH*DL * sizeof(float));
  float * w = (float *) malloc(DW*DH*DL * sizeof(float));
  IO::posix_read<float>(u_file_name.c_str(),u, DW*DH*DL);
  IO::posix_read<float>(v_file_name.c_str(),v, DW*DH*DL);
  IO::posix_read<float>(w_file_name.c_str(),w, DW*DH*DL);
  MPI_Barrier(MPI_COMM_WORLD);
	if(rank == 0){
		elapsed_time += MPI_Wtime();
		reading_time += elapsed_time;
	}
  if(rank == 0){
		cout << "Reading time: " << reading_time << endl;
	}
  int xn = DW;
  int yn = DH;
  int zn = DL;
  int compress_xn = DW;
  int compress_yn = DH;
  int compress_zn = DL;
  int front = 0, back = 0, top = 0, bot = 0, left = 0, right = 0;

  if (size == 1){
    bot = 1;
    top = 1;
    left = 1;
    right = 1;
    front = 1;
    back = 1;
    compress_xn = DW;
    compress_yn = DH;
    compress_zn = DL;
    xn = DW;
    yn = DH;
    zn = DL;
  }
  else{
  if (coords[0] == 0)
  {
    bot = 1;
    zn += 1;
    compress_zn += 1;
  }
  else if (coords[0] == size_cubert - 1)
  {
    top = 1;
    compress_zn += 1;
  }
  else{
    zn += 1;
    compress_zn += 2;
    }
  if (coords[1] == 0)
  {
    back = 1;
    yn += 1;
    compress_yn += 1;
  }
  else if (coords[1] == size_cubert - 1)
  {
    front = 1;
    compress_yn += 1;
  }
  else{
    yn += 1;
    compress_yn += 2;}

  if (coords[2] == 0)
  {
    left = 1;
    xn += 1;
    compress_xn += 1;
  }
  else if (coords[2] == size_cubert - 1)
  {
    right = 1;
    compress_xn += 1;
  }
  else{
    xn += 1;
    compress_xn += 2;}

  }
  
  size_t block_size = xn * yn * zn;
  size_t compress_block_size = compress_xn * compress_yn * compress_zn;
  // printf("block_size %d, compress_block_size %d,front %d,back %d,top %d,bot %d,left %d,right %d\n",block_size,compress_block_size,front,back,top,bot,left,right);
  float *new_U = (float *)malloc(sizeof(float) * block_size);
  float *new_V = (float *)malloc(sizeof(float) * block_size);
  float *new_W = (float *)malloc(sizeof(float) * block_size);
  cover_ghost_points(u,v,w,new_U,new_V,new_W,DL,DH,DW,xn,yn,zn,front,back,top,bot,left,right,coords,comm_cart,MPI_COMM_WORLD,status);


  const int type_bits = 63;
  double vector_field_resolution = 0;
  uint64_t vector_field_scaling_factor = 1;
  double min_val = 0;
  double GLOBAL_min_val = 0;
  double GLOBAL_vector_field_resolution = 0;


  for (size_t i = 0; i < xn*yn*zn; i++)
  {
    min_val = std::max(std::max(fabs(new_U[i]), fabs(new_V[i])), fabs(new_W[i]));
    vector_field_resolution = std::max(vector_field_resolution, min_val);
  }
  // std::cout << "MIN VALUE = " << min_val << std::endl;
  MPI_Allreduce(&min_val, &GLOBAL_min_val, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&vector_field_resolution, &GLOBAL_vector_field_resolution, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  vector_field_resolution = std::max(GLOBAL_vector_field_resolution, GLOBAL_min_val);
  // printf("rank %d: GLOBAL MAX value %f, global_vec_field_resol = %f , vec_field_resol = %f\n", rank, GLOBAL_min_val, GLOBAL_vector_field_resolution, vector_field_resolution);
  // std::cout << "GLOBAL MIN VAL = " << GLOBAL_min_val << std::endl;
  int vbits = std::ceil(std::log2(vector_field_resolution));
  int nbits = (type_bits - 5) / 3;
  vector_field_scaling_factor = 1 << (nbits - vbits);
  auto critical_points_0 = compute_critical_points(new_U, new_V, new_W, zn, yn, xn, vector_field_scaling_factor);
  free(new_U);
  free(new_V);
  free(new_W);

  // compute decompressed data
  size_t result_size = 0;
  double max_eb = atof(argv[7]);
  int option = atoi(argv[8]);
  if((option != 0) && (option != 1) && (option != 2)){
    printf("option should be 0 (no protection for borders), 1  (lossless borders), or 2 (optimized error bound derivation)\n");
    MPI_Finalize();
    exit(0);
  }
  size_t num_elements = DL*DH*DW;
  
  size_t GLOBAL_num_elements = 0;
  size_t GLOBAL_lossless_outsize;

  unsigned char *result = NULL;
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank == 0) elapsed_time = -MPI_Wtime();
  // printf("main: sz_compress_cp_preserve_sos_3d_online_fp\n");
  if(option == 0) result = sz_compress_cp_preserve_sos_3d_online_fp(u, v, w, DL, DH, DW, result_size, true, max_eb);
  else if(option == 1) result = sz_compress_cp_preserve_sos_3d_online_fp_parallel_lossless_border(u, v, w, DL, DH, DW, result_size, true, max_eb);
  else result = sz_compress_cp_preserve_sos_3d_online_fp_parallel(u, v, w, DL, DH, DW, result_size, false, max_eb);
  free(u);
  free(v);
  free(w);

  unsigned char *result_after_lossless = NULL;
  size_t lossless_outsize = sz_lossless_compress(ZSTD_COMPRESSOR, 3, result, result_size, &result_after_lossless);
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank == 0){
  elapsed_time += MPI_Wtime();
  compression_time += elapsed_time;
  }
  if(rank == 0) cout << "Compression time: " << compression_time << "s" << endl;

  MPI_Barrier(MPI_COMM_WORLD);
  if(rank == 0) elapsed_time = -MPI_Wtime();
  IO::posix_write((u_file_name + ".cpsz").c_str(), result_after_lossless, lossless_outsize);
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank == 0){
		elapsed_time += MPI_Wtime();
		write_compressed_data_time = elapsed_time;
	}
  if(rank == 0) cout << "Writing compressed time: " << write_compressed_data_time << "s" << endl;
  IO::clear_cache();
  double ratio = (3 * num_elements * sizeof(float)) * 1.0 / lossless_outsize;
  // printf("ID = %d, Num_elements = %d, lossless_outsize = %d, local ratio = %f\n", rank, num_elements, lossless_outsize, ratio);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Reduce(&num_elements, &GLOBAL_num_elements, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&lossless_outsize, &GLOBAL_lossless_outsize, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  
  //cout << "ID = " << rank << " ,Num_elements = " << num_elements << ", Compressed size = " << lossless_outsize << ", local ratio = " << ratio << endl;
  free(result);
  free(result_after_lossless);

  //read compressed data
  result_after_lossless = (unsigned char *)malloc(lossless_outsize);
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank == 0) elapsed_time = -MPI_Wtime();
  IO::posix_read<unsigned char>((u_file_name + ".cpsz").c_str(),result_after_lossless, lossless_outsize);
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank == 0){
    elapsed_time += MPI_Wtime();
    reading_compressed_data_time += elapsed_time;
  }
  if(rank == 0) {
    cout << "Reading compressed time: " << reading_compressed_data_time << "s" << endl;
  }

  // decompression
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank == 0) elapsed_time = -MPI_Wtime();
  size_t lossless_output = sz_lossless_decompress(ZSTD_COMPRESSOR, result_after_lossless, lossless_outsize, &result, result_size);
  float *dec_U = NULL;
  float *dec_V = NULL;
  float *dec_W = NULL;
  if((option == 0) || (option == 1)) sz_decompress_cp_preserve_3d_online_fp<float>(result, DL, DH, DW, dec_U, dec_V, dec_W);
  else sz_decompress_cp_preserve_3d_online_fp_parallel<float>(result, DL, DH, DW, dec_U, dec_V, dec_W);
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank == 0){
    elapsed_time += MPI_Wtime();
    decompression_time = elapsed_time;
  }
  if(rank == 0) {
    cout << "Decompression time: " << decompression_time << "s" << endl;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank == 0) elapsed_time = -MPI_Wtime();
  IO::posix_write((u_file_name + ".out").c_str(), dec_U, num_elements);
  IO::posix_write((v_file_name + ".out").c_str(), dec_V, num_elements);
  IO::posix_write((w_file_name + ".out").c_str(), dec_W, num_elements);
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank == 0){
    elapsed_time += MPI_Wtime();
    writing_time = elapsed_time;
  }
  if(rank == 0) {
    cout << "Writing time: " << writing_time << "s" << endl;
  }
  free(result_after_lossless);
  // now use dec_U,dec_V,dec_W to cover_ghost()

  float *new_dec_U = (float *)malloc(sizeof(float) * block_size);
  float *new_dec_V = (float *)malloc(sizeof(float) * block_size);
  float *new_dec_W = (float *)malloc(sizeof(float) * block_size);

  MPI_Barrier(MPI_COMM_WORLD);
  if(rank == 0) elapsed_time = -MPI_Wtime();
  cover_ghost_points(dec_U,dec_V,dec_W,new_dec_U,new_dec_V,new_dec_W,DL,DH,DW,xn,yn,zn,front,back,top,bot,left,right,coords,comm_cart,MPI_COMM_WORLD,status);
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank == 0){
		elapsed_time += MPI_Wtime();
		buid_ghost_point_time += elapsed_time;
	}
  if(rank == 0) {
    cout << "build ghost points time: " << buid_ghost_point_time << "s" << endl;
  }
  // END OF compute decompressed data
  auto critical_points_1 = compute_critical_points(new_dec_U, new_dec_V, new_dec_W, zn, yn, xn, vector_field_scaling_factor);


  int matches = 0;
  std::vector<critical_point_t> fp, fn, ft, m;
  std::vector<critical_point_t> origin;
  for (const auto &p : critical_points_0)
  {
    auto cp = p.second;
    origin.push_back(p.second);
    if (critical_points_1.find(p.first) != critical_points_1.end())
    {
      matches++;
      auto cp_1 = critical_points_1[p.first];
      // std::cout << "critical points in cell " << p.first << ": positions from (" << cp.x[0] << ", " << cp.x[1] << ", " << cp.x[2] << ") to (" << cp_1.x[0] << ", " << cp_1.x[1] << ", " << cp_1.x[2] << ")" << std::endl;
      if (cp.type != cp_1.type)
      {
        // std::cout << "Change from " << cp.type
        //   << " to " << cp_1.type <<
        //    ", positions from " << cp.x[0] << ", " << cp.x[1] << ", " << cp.x[2] << " to " << cp_1.x[0] << ", " << cp_1.x[1] << ", " << cp_1.x[2] << std::endl;
        // printf("Type change, position from %.6f %.6f %.6f to %.6f %.6f %.6f\n", cp.x[0], cp.x[1], cp.x[2], cp_1.x[0], cp_1.x[1], cp_1.x[2]);
        ft.push_back(cp_1);
      }
      else
        m.push_back(cp_1);
    }
    else
      fn.push_back(p.second);
    // std::cout << std::endl;
  }
  for (const auto &p : critical_points_1)
  {
    if (critical_points_0.find(p.first) == critical_points_0.end())
    {
      fp.push_back(p.second);
    }
  }


  // MPI_Barrier(MPI_COMM_WORLD);
  if(fp.size() != 0 || fn.size() != 0 || ft.size() != 0)
  {
    std::cout << "ERROR=========  " << std::endl;
    std::cout << "ID =  " << rank << std::endl;
    std::cout << "Ground truth = " << critical_points_0.size() << std::endl;
    std::cout << "TP = " << m.size() << std::endl;
    std::cout << "FP = " << fp.size() << std::endl;
    std::cout << "FN = " << fn.size() << std::endl;
    std::cout << "FT = " << ft.size() << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  int local_ground_truth = critical_points_0.size();
  int local_tp = m.size();
  int local_fp = fp.size();
  int local_fn = fn.size();
  int local_ft = ft.size();
  // reduce all the local result into global
  int global_ground_truth;
  int global_tp;
  int global_fp;
  int global_fn;
  int global_ft;
  double global_ratio;
  double global_compression_time;

  MPI_Reduce(&local_ground_truth, &global_ground_truth, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_tp, &global_tp, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_fp, &global_fp, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_fn, &global_fn, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_ft, &global_ft, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  // MPI_Reduce(&ratio, &global_ratio, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  // MPI_Reduce(&compression_time, &global_compression_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  if (rank == 0)
  {
    std::cout << "GLOBAL=======  " << std::endl;
    std::cout << "GLOBAL Ground truth = " << global_ground_truth << std::endl;
    std::cout << "GLOBAL  TP = " << global_tp << std::endl;
    std::cout << "GLOBAL  FP = " << global_fp << std::endl;
    std::cout << "GLOBAL  FN = " << global_fn << std::endl;
    std::cout << "GLOBAL  FT = " << global_ft << std::endl;
    std::cout << "GLOBAL Ratio= " << (double)((3 * GLOBAL_num_elements * sizeof(float)) * 1.0 / GLOBAL_lossless_outsize) << std::endl;
  }
  free(dec_U);
  free(new_dec_U);
  free(dec_V);
  free(new_dec_V);
  free(dec_W);
  free(new_dec_W);
  MPI_Finalize();
  return 0;
}