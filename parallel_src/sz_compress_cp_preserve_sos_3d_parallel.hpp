#include "sz_cp_preserve_utils.hpp"
#include "sz_compress_cp_preserve_3d.hpp"
#include "sz_def.hpp"
#include "sz_compression_utils.hpp"
#include <unordered_map>
#include <ftk/numeric/critical_point_type.hh>
#include <ftk/numeric/critical_point_test.hh>
#include <ftk/numeric/fixed_point.hh>
#include <ftk/numeric/gradient.hh>
#include <ftk/numeric/print.hh>
#include <mpi.h>

// offsets to get 24 adjacent simplex indices
// x -> z, fastest -> slowest dimensions
// current data would always be the last index, i.e. x[i][3]
static const int coordinates[24][4][3] = {
	// offset = 0, 0, 0
	{
		{0, 0, 1},
		{0, 1, 1},
		{1, 1, 1},
		{0, 0, 0}
	},
	{
		{0, 1, 0},
		{0, 1, 1},
		{1, 1, 1},
		{0, 0, 0}
	},
	{
		{0, 0, 1},
		{1, 0, 1},
		{1, 1, 1},
		{0, 0, 0}
	},
	{
		{1, 0, 0},
		{1, 0, 1},
		{1, 1, 1},
		{0, 0, 0}
	},
	{
		{0, 1, 0},
		{1, 1, 0},
		{1, 1, 1},
		{0, 0, 0}
	},
	{
		{1, 0, 0},
		{1, 1, 0},
		{1, 1, 1},
		{0, 0, 0}
	},
	// offset = -1, 0, 0
	{
		{0, 0, 0},
		{1, 0, 1},
		{1, 1, 1},
		{1, 0, 0}
	},
	{
		{0, 0, 0},
		{1, 1, 0},
		{1, 1, 1},
		{1, 0, 0}
	},
	// offset = 0, -1, 0
	{
		{0, 0, 0},
		{0, 1, 1},
		{1, 1, 1},
		{0, 1, 0}
	},
	{
		{0, 0, 0},
		{1, 1, 0},
		{1, 1, 1},
		{0, 1, 0}
	},
	// offset = -1, -1, 0
	{
		{0, 0, 0},
		{0, 1, 0},
		{1, 1, 1},
		{1, 1, 0}
	},
	{
		{0, 0, 0},
		{1, 0, 0},
		{1, 1, 1},
		{1, 1, 0}
	},
	// offset = 0, 0, -1
	{
		{0, 0, 0},
		{0, 1, 1},
		{1, 1, 1},
		{0, 0, 1}
	},
	{
		{0, 0, 0},
		{1, 0, 1},
		{1, 1, 1},
		{0, 0, 1}
	},
	// offset = -1, 0, -1
	{
		{0, 0, 0},
		{0, 0, 1},
		{1, 1, 1},
		{1, 0, 1}
	},
	{
		{0, 0, 0},
		{1, 0, 0},
		{1, 1, 1},
		{1, 0, 1}
	},
	// offset = 0, -1, -1
	{
		{0, 0, 0},
		{0, 0, 1},
		{1, 1, 1},
		{0, 1, 1}
	},
	{
		{0, 0, 0},
		{0, 1, 0},
		{1, 1, 1},
		{0, 1, 1}
	},
	// offset = -1, -1, -1
	{
		{0, 0, 0},
		{0, 0, 1},
		{0, 1, 1},
		{1, 1, 1}
	},
	{
		{0, 0, 0},
		{0, 1, 0},
		{0, 1, 1},
		{1, 1, 1}
	},
	{
		{0, 0, 0},
		{0, 0, 1},
		{1, 0, 1},
		{1, 1, 1}
	},
	{
		{0, 0, 0},
		{1, 0, 0},
		{1, 0, 1},
		{1, 1, 1}
	},
	{
		{0, 0, 0},
		{0, 1, 0},
		{1, 1, 0},
		{1, 1, 1}
	},
	{
		{0, 0, 0},
		{1, 0, 0},
		{1, 1, 0},
		{1, 1, 1}
	}
};

// default coordinates for tets in a cell
static const double default_coords[6][4][3] = {
  {
    {0, 0, 0},
    {0, 0, 1},
    {0, 1, 1},
    {1, 1, 1}
  },
  {
    {0, 0, 0},
    {0, 1, 0},
    {0, 1, 1},
    {1, 1, 1}
  },
  {
    {0, 0, 0},
    {0, 0, 1},
    {1, 0, 1},
    {1, 1, 1}
  },
  {
    {0, 0, 0},
    {1, 0, 0},
    {1, 0, 1},
    {1, 1, 1}
  },
  {
    {0, 0, 0},
    {0, 1, 0},
    {1, 1, 0},
    {1, 1, 1}
  },
  {
    {0, 0, 0},
    {1, 0, 0},
    {1, 1, 0},
    {1, 1, 1}
  },
};

// compute offsets for simplex, index, and positions
static void 
compute_offset(ptrdiff_t dim0_offset, ptrdiff_t dim1_offset, ptrdiff_t cell_dim0_offset, ptrdiff_t cell_dim1_offset,
				int simplex_offset[24], int index_offset[24][3][3], int offset[24][3]){
	int * simplex_offset_pos = simplex_offset;
	ptrdiff_t base = 0;
	// offset = 0, 0, 0
	for(int i=0; i<6; i++){
		*(simplex_offset_pos++) = i;
	}
	// offset = -1, 0, 0
	base = -6;
	*(simplex_offset_pos++) = base + 3;
	*(simplex_offset_pos++) = base + 5;
	// offset = 0, -1, 0
	base = -6*cell_dim1_offset;
	*(simplex_offset_pos++) = base + 1;
	*(simplex_offset_pos++) = base + 4;
	// offset = -1, -1, 0
	base = -6 - 6*cell_dim1_offset;
	*(simplex_offset_pos++) = base + 4;
	*(simplex_offset_pos++) = base + 5;
	// offset = 0, 0, -1
	base = -6*cell_dim0_offset;
	*(simplex_offset_pos++) = base + 0;
	*(simplex_offset_pos++) = base + 2;
	// offset = -1, 0, -1
	base = -6*cell_dim0_offset - 6;
	*(simplex_offset_pos++) = base + 2;
	*(simplex_offset_pos++) = base + 3;
	// offset = 0, -1, -1
	base = -6*cell_dim1_offset - 6*cell_dim0_offset;
	*(simplex_offset_pos++) = base + 0;
	*(simplex_offset_pos++) = base + 1;
	// offset = -1, -1, -1
	base = -6*cell_dim0_offset - 6*cell_dim1_offset - 6;
	for(int i=0; i<6; i++){
		*(simplex_offset_pos++) = base + i;
	}
	for(int i=0; i<24; i++){
		for(int j=0; j<3; j++){
			for(int k=0; k<3; k++){
				index_offset[i][j][k] = coordinates[i][j][k] - coordinates[i][3][k];
			}
		}
	}
	for(int i=0; i<24; i++){
		for(int x=0; x<3; x++){
			offset[i][x] = (coordinates[i][x][0] - coordinates[i][3][0]) + (coordinates[i][x][1] - coordinates[i][3][1]) * dim1_offset + (coordinates[i][x][2] - coordinates[i][3][2]) * dim0_offset;
		}
	}	
}

template<typename T_fp>
static int 
check_cp(T_fp vf[4][3], int indices[4]){
	// robust critical point test
	bool succ = ftk::robust_critical_point_in_simplex3(vf, indices);
	if (!succ) return -1;
	return 1;
}

#define SINGULAR 0
#define STABLE_SOURCE 1
#define UNSTABLE_SOURCE 2
#define STABLE_REPELLING_SADDLE 3
#define UNSTABLE_REPELLING_SADDLE 4
#define STABLE_ATRACTTING_SADDLE  5
#define UNSTABLE_ATRACTTING_SADDLE  6
#define STABLE_SINK 7
#define UNSTABLE_SINK 8

template<typename T>
static int
get_cp_type(const T X[4][3], const T U[4][3]){
	const T X_[3][3] = {
		{X[0][0] - X[3][0], X[1][0] - X[3][0], X[2][0] - X[3][0]}, 
		{X[0][1] - X[3][1], X[1][1] - X[3][1], X[2][1] - X[3][1]},
		{X[0][2] - X[3][2], X[1][2] - X[3][2], X[2][2] - X[3][2]}    
	};
	const T U_[3][3] = {
		{U[0][0] - U[3][0], U[1][0] - U[3][0], U[2][0] - U[3][0]}, 
		{U[0][1] - U[3][1], U[1][1] - U[3][1], U[2][1] - U[3][1]},
		{U[0][2] - U[3][2], U[1][2] - U[3][2], U[2][2] - U[3][2]}    
	};
	T inv_X_[3][3];
	ftk::matrix_inverse3x3(X_, inv_X_);
	T J[3][3];
	ftk::matrix3x3_matrix3x3_multiplication(inv_X_, U_, J);
	T P[4];
	ftk::characteristic_polynomial_3x3(J, P);
	std::complex<T> root[3];
	T disc = ftk::solve_cubic(P[2], P[1], P[0], root);
	if(fabs(disc) < std::numeric_limits<T>::epsilon()) return SINGULAR;
	int negative_real_parts = 0;
	for(int i=0; i<3; i++){
		negative_real_parts += (root[i].real() < 0);
	}
	switch(negative_real_parts){
		case 0:
			return (disc > 0) ? UNSTABLE_SOURCE : STABLE_SOURCE;
		case 1:
			return (disc > 0) ? UNSTABLE_REPELLING_SADDLE : STABLE_REPELLING_SADDLE;
		case 2:
			return (disc > 0) ? UNSTABLE_ATRACTTING_SADDLE : STABLE_ATRACTTING_SADDLE;
		case 3:
			return (disc > 0) ? UNSTABLE_SINK : STABLE_SINK;
		default:
			return SINGULAR;
	}
}

template<typename T_fp>
static int 
check_cp_type(const T_fp vf[4][3], const double v[4][3], const double X[4][3], const int indices[4]){
	// robust critical point test
	bool succ = ftk::robust_critical_point_in_simplex3(vf, indices);
	if (!succ) return -1;
	return get_cp_type(X, v);
}

template<typename T_fp_acc, typename T_fp>
static inline void 
update_index(T_fp_acc vf[4][3], int indices[4], int local_id, int global_id, const T_fp * U, const T_fp * V, const T_fp * W){
	indices[local_id] = global_id;
	vf[local_id][0] = U[global_id];
	vf[local_id][1] = V[global_id];
	vf[local_id][2] = W[global_id];
}

template<typename T_fp>
static vector<bool> 
compute_cp(const T_fp * U_fp, const T_fp * V_fp, const T_fp * W_fp, int r1, int r2, int r3){
	// check cp for all cells
	vector<bool> cp_exist(6*(r1-1)*(r2-1)*(r3-1), 0);
	ptrdiff_t dim0_offset = r2*r3;
	ptrdiff_t dim1_offset = r3;
	ptrdiff_t cell_dim0_offset = (r2-1)*(r3-1);
	ptrdiff_t cell_dim1_offset = r3-1;
	int indices[4] = {0};
	// __int128 vf[4][3] = {0};
	int64_t vf[4][3] = {0};
	for(int i=0; i<r1-1; i++){
		for(int j=0; j<r2-1; j++){
			for(int k=0; k<r3-1; k++){
				bool verbose = false;
				// order (reserved, z->x):
				ptrdiff_t cell_offset = 6*(i*cell_dim0_offset + j*cell_dim1_offset + k);
				ptrdiff_t tmp_offset = 6*(i*dim0_offset + j*dim1_offset + k);
				// if(tmp_offset/6 == 4001374/6) verbose = true;
				// if(verbose){
				// 	std::cout << i << " " << j << " " << k << std::endl;
				// }
				// (ftk-0) 000, 001, 011, 111
				update_index(vf, indices, 0, i*dim0_offset + j*dim1_offset + k, U_fp, V_fp, W_fp);
				update_index(vf, indices, 1, (i+1)*dim0_offset + j*dim1_offset + k, U_fp, V_fp, W_fp);
				update_index(vf, indices, 2, (i+1)*dim0_offset + (j+1)*dim1_offset + k, U_fp, V_fp, W_fp);
				update_index(vf, indices, 3, (i+1)*dim0_offset + (j+1)*dim1_offset + (k+1), U_fp, V_fp, W_fp);
				cp_exist[cell_offset] = (check_cp(vf, indices) == 1);
				// if(verbose){
				// 	auto offset = cell_offset;
				// 	std::cout << "cell id = " << offset << ", cp = " << +cp_exist[offset] << std::endl;
				// 	std::cout << "indices: ";
				// 	for(int i=0; i<4; i++){
				// 		std::cout << indices[i] << " "; 
				// 	}
				// 	std::cout << std::endl;
				// 	T_fp tmp[4][3];
				// 	for(int i=0; i<4; i++){
				// 		for(int j=0; j<3; j++){
				// 			tmp[i][j] = vf[i][j];
				// 		}
				// 	}
				// 	ftk::print4x3("M:", tmp);
				// }				
				// (ftk-2) 000, 010, 011, 111
				update_index(vf, indices, 1, i*dim0_offset + (j+1)*dim1_offset + k, U_fp, V_fp, W_fp);
				cp_exist[cell_offset + 1] = (check_cp(vf, indices) == 1);
				// (ftk-1) 000, 001, 101, 111
				update_index(vf, indices, 1, (i+1)*dim0_offset + j*dim1_offset + k, U_fp, V_fp, W_fp);
				update_index(vf, indices, 2, (i+1)*dim0_offset + j*dim1_offset + k+1, U_fp, V_fp, W_fp);
				cp_exist[cell_offset + 2] = (check_cp(vf, indices) == 1);
				// (ftk-4) 000, 100, 101, 111
				update_index(vf, indices, 1, i*dim0_offset + j*dim1_offset + k+1, U_fp, V_fp, W_fp);
				cp_exist[cell_offset + 3] = (check_cp(vf, indices) == 1);
				// if(verbose){
				// 	auto offset = cell_offset + 3;
				// 	std::cout << "cell id = " << offset << ", cp = " << +cp_exist[offset] << std::endl;
				// 	std::cout << "indices: ";
				// 	for(int i=0; i<4; i++){
				// 		std::cout << indices[i] << " "; 
				// 	}
				// 	std::cout << std::endl;
				// 	T_fp tmp[4][3];
				// 	for(int i=0; i<4; i++){
				// 		for(int j=0; j<3; j++){
				// 			tmp[i][j] = vf[i][j];
				// 		}
				// 	}

				// 	ftk::print4x3("M:", tmp);
				// }
				// (ftk-3) 000, 010, 110, 111
				update_index(vf, indices, 1, i*dim0_offset + (j+1)*dim1_offset + k, U_fp, V_fp, W_fp);
				update_index(vf, indices, 2, i*dim0_offset + (j+1)*dim1_offset + k+1, U_fp, V_fp, W_fp);
				cp_exist[cell_offset + 4] = (check_cp(vf, indices) == 1);
				// if(verbose){
				// 	auto offset = cell_offset + 4;
				// 	std::cout << "cell id = " << offset << ", cp = " << +cp_exist[offset] << std::endl;
				// 	std::cout << "indices: ";
				// 	for(int i=0; i<4; i++){
				// 		std::cout << indices[i] << " "; 
				// 	}
				// 	std::cout << std::endl;
				// 	T_fp tmp[4][3];
				// 	for(int i=0; i<4; i++){
				// 		for(int j=0; j<3; j++){
				// 			tmp[i][j] = vf[i][j];
				// 		}
				// 	}
				// 	ftk::print4x3("M:", tmp);
				// }				
				// (ftk-5) 000, 100, 110, 111
				update_index(vf, indices, 1, i*dim0_offset + j*dim1_offset + k+1, U_fp, V_fp, W_fp);
				cp_exist[cell_offset + 5] = (check_cp(vf, indices) == 1);
			}
		}
	}
	return cp_exist;	
}

template<typename T_data>
static inline void 
update_value(double v[4][3], int local_id, int global_id, const T_data * U, const T_data * V, const T_data * W){
	v[local_id][0] = U[global_id];
	v[local_id][1] = V[global_id];
	v[local_id][2] = W[global_id];
}

template<typename T_data, typename T_fp>
static vector<int> 
compute_cp_and_type(const T_fp * U_fp, const T_fp * V_fp, const T_fp * W_fp, const T_data * U, const T_data * V, const T_data * W, int r1, int r2, int r3){
	// check cp for all cells
	vector<int> cp_type(6*(r1-1)*(r2-1)*(r3-1), 0);
	ptrdiff_t dim0_offset = r2*r3;
	ptrdiff_t dim1_offset = r3;
	ptrdiff_t cell_dim0_offset = (r2-1)*(r3-1);
	ptrdiff_t cell_dim1_offset = r3-1;
	int indices[4] = {0};
	// __int128 vf[4][3] = {0};
	int64_t vf[4][3] = {0};
	double v[4][3] = {0};
	for(int i=0; i<r1-1; i++){
		for(int j=0; j<r2-1; j++){
			for(int k=0; k<r3-1; k++){
				bool verbose = false;
				// order (reserved, z->x):
				ptrdiff_t cell_offset = 6*(i*cell_dim0_offset + j*cell_dim1_offset + k);
				ptrdiff_t tmp_offset = 6*(i*dim0_offset + j*dim1_offset + k);
				// (ftk-0) 000, 001, 011, 111
				update_index(vf, indices, 0, i*dim0_offset + j*dim1_offset + k, U_fp, V_fp, W_fp);
				update_index(vf, indices, 1, (i+1)*dim0_offset + j*dim1_offset + k, U_fp, V_fp, W_fp);
				update_index(vf, indices, 2, (i+1)*dim0_offset + (j+1)*dim1_offset + k, U_fp, V_fp, W_fp);
				update_index(vf, indices, 3, (i+1)*dim0_offset + (j+1)*dim1_offset + (k+1), U_fp, V_fp, W_fp);
				for(int p=0; p<4; p++){
					v[p][0] = U[indices[p]];
					v[p][1] = V[indices[p]];
					v[p][2] = W[indices[p]];
				}
				cp_type[cell_offset] = check_cp_type(vf, v, default_coords[0], indices);
				// (ftk-2) 000, 010, 011, 111
				update_index(vf, indices, 1, i*dim0_offset + (j+1)*dim1_offset + k, U_fp, V_fp, W_fp);
				update_value(v, 1, indices[1], U, V, W); 
				cp_type[cell_offset + 1] = check_cp_type(vf, v, default_coords[1], indices);
				// (ftk-1) 000, 001, 101, 111
				update_index(vf, indices, 1, (i+1)*dim0_offset + j*dim1_offset + k, U_fp, V_fp, W_fp);
				update_value(v, 1, indices[1], U, V, W); 
				update_index(vf, indices, 2, (i+1)*dim0_offset + j*dim1_offset + k+1, U_fp, V_fp, W_fp);
				update_value(v, 2, indices[2], U, V, W); 
				cp_type[cell_offset + 2] = check_cp_type(vf, v, default_coords[2], indices);
				// (ftk-4) 000, 100, 101, 111
				update_index(vf, indices, 1, i*dim0_offset + j*dim1_offset + k+1, U_fp, V_fp, W_fp);
				update_value(v, 1, indices[1], U, V, W); 
				cp_type[cell_offset + 3] = check_cp_type(vf, v, default_coords[3], indices);
				// (ftk-3) 000, 010, 110, 111
				update_index(vf, indices, 1, i*dim0_offset + (j+1)*dim1_offset + k, U_fp, V_fp, W_fp);
				update_value(v, 1, indices[1], U, V, W); 
				update_index(vf, indices, 2, i*dim0_offset + (j+1)*dim1_offset + k+1, U_fp, V_fp, W_fp);
				update_value(v, 2, indices[2], U, V, W); 
				cp_type[cell_offset + 4] = check_cp_type(vf, v, default_coords[4], indices);
				// (ftk-5) 000, 100, 110, 111
				update_index(vf, indices, 1, i*dim0_offset + j*dim1_offset + k+1, U_fp, V_fp, W_fp);
				update_value(v, 1, indices[1], U, V, W); 
				cp_type[cell_offset + 5] = check_cp_type(vf, v, default_coords[5], indices);
			}
		}
	}
	return cp_type;	
}

template <typename T> 
static bool 
same_direction(T u0, T u1, T u2, T u3) {
    int sgn0 = sgn(u0);
    if(sgn0 == 0) return false;
    if((sgn0 == sgn(u1)) && (sgn0 == sgn(u2)) && (sgn0 == sgn(u3))) return true;
    return false;
}

template<typename T>
static inline T
det_2_by_2(const T u0, const T u1, const T v0, const T v1){
	return u0*v1 - u1*v0;
}

template<typename T>
static inline T
det_3_by_3(const T u0, const T u1, const T u2, const T v0, const T v1, const T v2, const T w0, const T w1, const T w2){
	return u0*v1*w2 + u1*v2*w0 + u2*v0*w1 - u0*v2*w1 - u1*v0*w2 -u2*v1*w0;
}

template<typename T>
static inline T
abs(const T u){
	return (u>0) ? u : -u;
}

std::ostream& operator<<(std::ostream& o, const __int128& x) {
    if (x == std::numeric_limits<__int128>::min()) return o << "-170141183460469231731687303715884105728";
    if (x < 0) return o << "-" << -x;
    if (x < 10) return o << (char)(x + '0');
    return o << x / 10 << (char)(x % 10 + '0');
}
/*
Tet x0, x1, x2, x3, derive cp-preserving eb for x3 given x0, x1, x2
using SoS method
*/
template<typename T>
static T 
derive_cp_abs_eb_sos_online(const T u0, const T u1, const T u2, const T u3, const T v0, const T v1, const T v2, const T v3, const T w0, const T w1, const T w2, const T w3, bool verbose=false){
	T M0 = - det_3_by_3(u1, u2, u3, v1, v2, v3, w1, w2, w3);
	T M1 = + det_3_by_3(u0, u2, u3, v0, v2, v3, w0, w2, w3);
	T M2 = - det_3_by_3(u0, u1, u3, v0, v1, v3, w0, w1, w3);
	T M3 = + det_3_by_3(u0, u1, u2, v0, v1, v2, w0, w1, w2);
	T M = M0 + M1 + M2 + M3;
	if(M == 0) return 0;
	T same_eb = 0;
	if(same_direction(u0, u1, u2, u3)){			
		same_eb = MAX(same_eb, std::abs(u3));
	}
	if(same_direction(v0, v1, v2, v3)){			
		same_eb = MAX(same_eb, std::abs(v3));
	}
	if(same_direction(w0, w1, w2, w3)){			
		same_eb = MAX(same_eb, std::abs(w3));
	}
	if(same_eb != 0) return same_eb;
	// keep sign for the original simplex
	T one = 1;
	T denominator = abs(det_3_by_3(v0, v1, v2, w0, w1, w2, one, one, one)) + abs(det_3_by_3(u0, u1, u2, w0, w1, w2, one, one, one)) + abs(det_3_by_3(u0, u1, u2, v0, v1, v2, one, one, one)); 
	T eb = abs(M) / denominator;
	{
		// keep sign for replacing the three other vertices
		denominator = std::abs(det_2_by_2(v1, v2, w1, w2)) + std::abs(det_2_by_2(u1, u2, w1, w2)) + std::abs(det_2_by_2(u1, u2, v1, v2));
		if(denominator != 0){
			eb = MINF(eb, abs(M0) / denominator);
		}
		else return 0;
		denominator = std::abs(det_2_by_2(v0, v2, w0, w2)) + std::abs(det_2_by_2(u0, u2, w0, w2)) + std::abs(det_2_by_2(u0, u2, v0, v2));
		if(denominator != 0){
			eb = MINF(eb, abs(M1) / denominator);
		}
		else return 0;
		denominator = std::abs(det_2_by_2(v0, v1, w0, w1)) + std::abs(det_2_by_2(u0, u1, w0, w1)) + std::abs(det_2_by_2(u0, u1, v0, v1));
		if(denominator != 0){
			eb = MINF(eb, abs(M2) / denominator);
		}
		else return 0;
		// T cur_eb_0 = abs(M0)/(std::abs(det_2_by_2(v1, v2, w1, w2)) + std::abs(det_2_by_2(u1, u2, w1, w2)) + std::abs(det_2_by_2(u1, u2, v1, v2)));
		// T cur_eb_1 = abs(M1)/(std::abs(det_2_by_2(v0, v2, w0, w2)) + std::abs(det_2_by_2(u0, u2, w0, w2)) + std::abs(det_2_by_2(u0, u2, v0, v2)));
		// T cur_eb_2 = abs(M2)/(std::abs(det_2_by_2(v0, v1, w0, w1)) + std::abs(det_2_by_2(u0, u1, w0, w1)) + std::abs(det_2_by_2(u0, u1, v0, v1)));
		// eb = MINF(MINF(cur_eb_0, cur_eb_1), MINF(cur_eb_2, eb));
	}
	return eb;
}

template<typename T_acc, typename T>
static T 
derive_cp_abs_eb_sos_online_acc(const T u0, const T u1, const T u2, const T u3, const T v0, const T v1, const T v2, const T v3, const T w0, const T w1, const T w2, const T w3, bool verbose=false){
	T_acc M0 = - det_3_by_3<T_acc>(u1, u2, u3, v1, v2, v3, w1, w2, w3);
	// if(verbose){
	// 	std::cout << "M0 = " << M0 << std::endl;
	// }
	T_acc M1 = + det_3_by_3<T_acc>(u0, u2, u3, v0, v2, v3, w0, w2, w3);
	// if(verbose){
	// 	std::cout << "M1 = " << M1 << std::endl;
	// }
	T_acc M2 = - det_3_by_3<T_acc>(u0, u1, u3, v0, v1, v3, w0, w1, w3);
	// if(verbose){
	// 	std::cout << "M2 = " << M2 << std::endl;
	// }
	T_acc M3 = + det_3_by_3<T_acc>(u0, u1, u2, v0, v1, v2, w0, w1, w2);
	// if(verbose){
	// 	std::cout << "M3 = " << M3 << std::endl;
	// }
	T_acc M = M0 + M1 + M2 + M3;
	// if(verbose){
	// 	std::cout << u0 << " " << v0 << " " << w0 << std::endl;
	// 	std::cout << u1 << " " << v1 << " " << w1 << std::endl;
	// 	std::cout << u2 << " " << v2 << " " << w2 << std::endl;
	// 	std::cout << u3 << " " << v3 << " " << w3 << std::endl;
	// 	std::cout << "det = " << M << ": " << M0 << " " << M1 << " " << M2 << " " << M3 << std::endl;
	// }
	if(M == 0) return 0;
	T same_eb = 0;
	if(same_direction(u0, u1, u2, u3)){			
		same_eb = MAX(same_eb, std::abs(u3));
	}
	if(same_direction(v0, v1, v2, v3)){			
		same_eb = MAX(same_eb, std::abs(v3));
	}
	if(same_direction(w0, w1, w2, w3)){			
		same_eb = MAX(same_eb, std::abs(w3));
	}
	// if(verbose){
	// 	std::cout << "same_eb = " << same_eb << std::endl;
	// }
	if(same_eb != 0) return same_eb;
	// keep sign for the original simplex
	T one = 1;
	T_acc denominator = abs(det_3_by_3(v0, v1, v2, w0, w1, w2, one, one, one)) + abs(det_3_by_3(u0, u1, u2, w0, w1, w2, one, one, one)) + abs(det_3_by_3(u0, u1, u2, v0, v1, v2, one, one, one)); 
	T_acc eb = abs(M) / denominator;
	{
		// keep sign for replacing the three other vertices
		T_acc cur_eb_0 = abs(M0)/(std::abs(det_2_by_2(v1, v2, w1, w2)) + std::abs(det_2_by_2(u1, u2, w1, w2)) + std::abs(det_2_by_2(u1, u2, v1, v2)));
		T_acc cur_eb_1 = abs(M1)/(std::abs(det_2_by_2(v0, v2, w0, w2)) + std::abs(det_2_by_2(u0, u2, w0, w2)) + std::abs(det_2_by_2(u0, u2, v0, v2)));
		T_acc cur_eb_2 = abs(M2)/(std::abs(det_2_by_2(v0, v1, w0, w1)) + std::abs(det_2_by_2(u0, u1, w0, w1)) + std::abs(det_2_by_2(u0, u1, v0, v1)));
		// if(verbose){
		// 	T d1 = det_2_by_2(v1, v2, w1, w2);
		// 	T d2 = det_2_by_2(u1, u2, w1, w2);
		// 	T d3 = det_2_by_2(u1, u2, v1, v2);
		// 	std::cout << "denominator 1 = " << d1 << " " << d2 << " " << d3 << std::endl;
		// 	std::cout << "M0 = " << (T_acc) u3 * d1 - (T_acc) v3 * d2 + (T_acc) w3 * d3 << std::endl;
		// 	std::cout << "eb 1-3 = " << cur_eb_0 << " " << cur_eb_1 << " " << cur_eb_2 << std::endl;
		// }
		eb = MINF(MINF(cur_eb_0, cur_eb_1), MINF(cur_eb_2, eb));
	}
	return (T) MINF(eb, (T_acc) std::numeric_limits<int64_t>::max());
}

template<typename T, typename T_fp>
static int64_t 
convert_to_fixed_point(const T * U, const T * V, const T * W, size_t num_elements, T_fp * U_fp, T_fp * V_fp, T_fp * W_fp, T_fp& range, int type_bits=63){
	double vector_field_resolution = 0;
	int64_t vector_field_scaling_factor = 1;
	for (int i=0; i<num_elements; i++){
		double min_val = std::max(std::max(fabs(U[i]), fabs(V[i])), fabs(W[i]));
		vector_field_resolution = std::max(vector_field_resolution, min_val);
	}
	int vbits = std::ceil(std::log2(vector_field_resolution));
	// uncomment when test MPI version
	MPI_Allreduce(MPI_IN_PLACE, &vbits, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
	int nbits = (type_bits - 5) / 3;
	vector_field_scaling_factor = 1 << (nbits - vbits);
	// std::cerr << "resolution=" << vector_field_resolution 
	// << ", factor=" << vector_field_scaling_factor 
	// << ", nbits=" << nbits << ", vbits=" << vbits << ", shift_bits=" << nbits - vbits << std::endl;
	int64_t max = std::numeric_limits<int64_t>::min();
	int64_t min = std::numeric_limits<int64_t>::max();
	// printf("max = %lld, min = %lld\n", max, min);
	for(int i=0; i<num_elements; i++){
		U_fp[i] = U[i] * vector_field_scaling_factor;
		V_fp[i] = V[i] * vector_field_scaling_factor;
		W_fp[i] = W[i] * vector_field_scaling_factor;
		max = std::max(max, U_fp[i]);
		max = std::max(max, V_fp[i]);
		max = std::max(max, W_fp[i]);
		min = std::min(min, U_fp[i]);
		min = std::min(min, V_fp[i]);
		min = std::min(min, W_fp[i]);
	}
	// printf("max = %lld, min = %lld\n", max, min);
	range = max - min;
	MPI_Allreduce(MPI_IN_PLACE, &range, 1, MPI_LONG, MPI_MIN, MPI_COMM_WORLD);
	return vector_field_scaling_factor;
}

void cover_ghost_points_bot_back_left(const float *u, const float *v, const float *w, float *new_U, float *new_V, float *new_W,
                        int DL, int DH, int DW,int compress_xn, int compress_yn, int compress_zn, int front, int back, int top, int bot, int left, int right,
                        int *coords, MPI_Comm comm_cart,MPI_Comm mpi_comm, MPI_Status status)
{

  // 1. 拷贝原始值，到新的cube中间
  const float *u_pos = u;
  const float *v_pos = v;
  const float *w_pos = w;
  for (size_t z = 0; z < DL; z++)
  {
    for (size_t y = 0; y < DH; y++)
    {

      memcpy(new_U + (z+(bot ^ 1)) * (compress_xn * compress_yn) + (y+(back^1)) * compress_xn + (left ^ 1), u_pos, DW * sizeof(float));
      memcpy(new_V + (z+(bot ^ 1)) * (compress_xn * compress_yn) + (y+(back^1)) * compress_xn + (left ^ 1), v_pos, DW * sizeof(float));
      memcpy(new_W + (z+(bot ^ 1)) * (compress_xn * compress_yn) + (y+(back^1)) * compress_xn + (left ^ 1), w_pos, DW * sizeof(float));
      u_pos += DW;
      v_pos += DW;
      w_pos += DW;
    }
  }
	// printf("Rank (%d, %d, %d) start ghost elements\n", coords[0], coords[1], coords[2]);
  // 2. 先贴3个面，左，后，下
  // 先处理的三个面改成相反的三个面了！
  // 发送top 面
  if (top != 1)
  {
    float *top_surface_u = (float *)malloc(sizeof(float) * DW * DH); // N*N,
    float *top_surface_v = (float *)malloc(sizeof(float) * DW * DH);
    float *top_surface_w = (float *)malloc(sizeof(float) * DW * DH);
    memcpy(top_surface_u, u + (DL - 1) * DW * DH, sizeof(float) * DW * DH); // 将u的值拷贝到top_surface_u中
    memcpy(top_surface_v, v + (DL - 1) * DW * DH, sizeof(float) * DW * DH);
    memcpy(top_surface_w, w + (DL - 1) * DW * DH, sizeof(float) * DW * DH);
    int top_coords[3] = {coords[0] + 1, coords[1], coords[2]}; // 找到上面block的coord
    int top_rank;
    MPI_Cart_rank(comm_cart, top_coords, &top_rank);
    MPI_Send(top_surface_u, DW * DH, MPI_FLOAT, top_rank, 0, MPI_COMM_WORLD);
    MPI_Send(top_surface_v, DW * DH, MPI_FLOAT, top_rank, 1, MPI_COMM_WORLD);
    MPI_Send(top_surface_w, DW * DH, MPI_FLOAT, top_rank, 2, MPI_COMM_WORLD);
    free(top_surface_u);
    free(top_surface_v);
    free(top_surface_w);
  }

  if (bot != 1)
  {
    // 接收并copy到bot面
    float *recvied_from_bot_u = (float *)malloc(sizeof(float) * DW * DH);
    float *recvied_from_bot_v = (float *)malloc(sizeof(float) * DW * DH);
    float *recvied_from_bot_w = (float *)malloc(sizeof(float) * DW * DH);
    float *recvied_from_bot_u_pos = recvied_from_bot_u;
    float *recvied_from_bot_v_pos = recvied_from_bot_v;
    float *recvied_from_bot_w_pos = recvied_from_bot_w;
    // 接收来自下方block的数据
    int bot_coords[3] = {coords[0] - 1, coords[1], coords[2]}; // 找到上面block的coord
    int bot_rank;
    MPI_Cart_rank(comm_cart, bot_coords, &bot_rank);
    MPI_Recv(recvied_from_bot_u_pos, DW * DH, MPI_FLOAT, bot_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(recvied_from_bot_v_pos, DW * DH, MPI_FLOAT, bot_rank, 1, MPI_COMM_WORLD, &status);
    MPI_Recv(recvied_from_bot_w_pos, DW * DH, MPI_FLOAT, bot_rank, 2, MPI_COMM_WORLD, &status);
    // 将接收到的数据按特定顺序拷贝到new_U中
    for (size_t i = 0; i < DH; i++)
    {
      // printf("i = %d",i);
      size_t offset = (back ^ 1) * compress_xn + (left ^ 1);
      memcpy(new_U + offset + i * compress_xn, recvied_from_bot_u_pos, sizeof(float) * DW);
      memcpy(new_V + offset + i * compress_xn, recvied_from_bot_v_pos, sizeof(float) * DW);
      memcpy(new_W + offset + i * compress_xn, recvied_from_bot_w_pos, sizeof(float) * DW);
      recvied_from_bot_u_pos += DW;
      recvied_from_bot_v_pos += DW;
      recvied_from_bot_w_pos += DW;
    }
    free(recvied_from_bot_u);
    free(recvied_from_bot_v);
    free(recvied_from_bot_w);
  }
	// printf("Rank (%d, %d, %d) finish top\n", coords[0], coords[1], coords[2]);

  //////////////////////////
  // 发送right面
  if (right != 1)
  {
    // 发送right面到右边的block
    float *right_surface_u = (float *)malloc(sizeof(float) * DH * DL); // N*N
    float *right_surface_v = (float *)malloc(sizeof(float) * DH * DL); // N*N
    float *right_surface_w = (float *)malloc(sizeof(float) * DH * DL); // N*N
    float *right_surface_u_pos = right_surface_u;
    float *right_surface_v_pos = right_surface_v;
    float *right_surface_w_pos = right_surface_w;
    for (size_t i = DW-1; i < DW * DH * DL; i += DW)
    {
      *right_surface_u_pos = u[i];
      right_surface_u_pos++;
      *right_surface_v_pos = v[i];
      right_surface_v_pos++;
      *right_surface_w_pos = w[i];
      right_surface_w_pos++;
    }
    int right_coords[3] = {coords[0], coords[1], coords[2] + 1}; // 找到右面block的coord
    int right_rank;
    MPI_Cart_rank(comm_cart, right_coords, &right_rank);
    MPI_Send(right_surface_u, DH * DL, MPI_FLOAT, right_rank, 3, MPI_COMM_WORLD);
    MPI_Send(right_surface_v, DH * DL, MPI_FLOAT, right_rank, 4, MPI_COMM_WORLD);
    MPI_Send(right_surface_w, DH * DL, MPI_FLOAT, right_rank, 5, MPI_COMM_WORLD);

    free(right_surface_u);
    free(right_surface_v);
    free(right_surface_w);
  }
  if (left != 1)
  {
    // 接收来自左边block的数据
    int left_coords[3] = {coords[0], coords[1], coords[2] - 1}; // 找到左面block的coord
    int left_rank;
    float *recvied_from_left_u = (float *)malloc(sizeof(float) * DH * DL);
    float *recvied_from_left_v = (float *)malloc(sizeof(float) * DH * DL);
    float *recvied_from_left_w = (float *)malloc(sizeof(float) * DH * DL);
    MPI_Cart_rank(comm_cart, left_coords, &left_rank);
    MPI_Recv(recvied_from_left_u, DH * DL, MPI_FLOAT, left_rank, 3, MPI_COMM_WORLD, &status);
    MPI_Recv(recvied_from_left_v, DH * DL, MPI_FLOAT, left_rank, 4, MPI_COMM_WORLD, &status);
    MPI_Recv(recvied_from_left_w, DH * DL, MPI_FLOAT, left_rank, 5, MPI_COMM_WORLD, &status);

    float *recvied_from_left_u_pos = recvied_from_left_u;
    float *recvied_from_left_v_pos = recvied_from_left_v;
    float *recvied_from_left_w_pos = recvied_from_left_w;
    // 将接收到的数据按特定顺序拷贝到new_U中
    for (size_t z = 0; z < DL; z++)
    {
      for (size_t y = 0; y < DH; y++)
      {
        size_t offset_z = (z + (bot ^ 1)) * compress_xn * compress_yn;
        size_t offset_y = (y + (back ^ 1)) * compress_xn;
        new_U[offset_y + offset_z] = *recvied_from_left_u_pos;
		recvied_from_left_u_pos++;
        new_V[offset_y + offset_z] = *recvied_from_left_v_pos;
		recvied_from_left_v_pos++;
        new_W[offset_y + offset_z] = *recvied_from_left_w_pos;
		recvied_from_left_w_pos++;
        // new_U[(xn - 1) + y * xn + z * xn * yn] = *recvied_from_right_u_pos++;
        // new_V[(xn - 1) + y * xn + z * xn * yn] = *recvied_from_right_v_pos++;
        // new_W[(xn - 1) + y * xn + z * xn * yn] = *recvied_from_right_w_pos++;
      }
    }
    free(recvied_from_left_u);
    free(recvied_from_left_v);
    free(recvied_from_left_w);
  }
  //////////////////////////
	// printf("Rank (%d, %d, %d) finish right\n", coords[0], coords[1], coords[2]);

  // 发送front面
  if (front != 1)
  {
    // 发送front面数据到前面的block
    float *front_surface_u = (float *)malloc(sizeof(float) * DL * DW); // N*N
    float *front_surface_v = (float *)malloc(sizeof(float) * DL * DW); // N*N
    float *front_surface_w = (float *)malloc(sizeof(float) * DL * DW); // N*N
    float *front_surface_u_pos = front_surface_u;
    float *front_surface_v_pos = front_surface_v;
    float *front_surface_w_pos = front_surface_w;
    for (size_t i = 0; i < DW * DH * DL; i += DW * DH)
    {
      size_t offset =  DW * DH - DW;
      memcpy(front_surface_u_pos, u + offset + i, sizeof(float) * DW);
      front_surface_u_pos += DW;
      memcpy(front_surface_v_pos, v + offset + i, sizeof(float) * DW);
      front_surface_v_pos += DW;
      memcpy(front_surface_w_pos, w + offset + i, sizeof(float) * DW);
      front_surface_w_pos += DW;
    }
    int front_coords[3] = {coords[0], coords[1] + 1, coords[2]}; // 找到front面block的coord
    int front_rank;
    MPI_Cart_rank(comm_cart, front_coords, &front_rank);
    MPI_Send(front_surface_u, DL * DW, MPI_FLOAT, front_rank, 6, MPI_COMM_WORLD);
    MPI_Send(front_surface_v, DL * DW, MPI_FLOAT, front_rank, 7, MPI_COMM_WORLD);
    MPI_Send(front_surface_w, DL * DW, MPI_FLOAT, front_rank, 8, MPI_COMM_WORLD);
    free(front_surface_u);
    free(front_surface_v);
    free(front_surface_w);
  }
  if (back != 1)
  {
    // 接收来自后方block的数据到back面
    float *recvied_from_back_u = (float *)malloc(sizeof(float) * DL * DW);
    float *recvied_from_back_v = (float *)malloc(sizeof(float) * DL * DW);
    float *recvied_from_back_w = (float *)malloc(sizeof(float) * DL * DW);

    int back_coords[3] = {coords[0], coords[1] - 1, coords[2]}; // 找到后面block的coord
    int back_rank;
    MPI_Cart_rank(comm_cart, back_coords, &back_rank);
    MPI_Recv(recvied_from_back_u, DL * DW, MPI_FLOAT, back_rank, 6, MPI_COMM_WORLD, &status);
    MPI_Recv(recvied_from_back_v, DL * DW, MPI_FLOAT, back_rank, 7, MPI_COMM_WORLD, &status);
    MPI_Recv(recvied_from_back_w, DL * DW, MPI_FLOAT, back_rank, 8, MPI_COMM_WORLD, &status);

    float *recvied_from_back_u_pos = recvied_from_back_u;
    float *recvied_from_back_v_pos = recvied_from_back_v;
    float *recvied_from_back_w_pos = recvied_from_back_w;
    // 将接收到的数据按特定顺序拷贝到new_U中
    for (size_t z = 0; z < DL; z++)
    {
      //size_t offset = xn * yn - xn;
      size_t z_offset = (bot ^ 1) * compress_xn * compress_yn; //z平面offset
      size_t offset =  (left ^ 1); //xy平面offset
      memcpy(new_U + z_offset + offset + (z * compress_xn * compress_yn), recvied_from_back_u_pos, DW * sizeof(float));
      memcpy(new_V + z_offset + offset + (z * compress_xn * compress_yn), recvied_from_back_v_pos, DW * sizeof(float));
      memcpy(new_W + z_offset + offset + (z * compress_xn * compress_yn), recvied_from_back_w_pos, DW * sizeof(float));

      // memcpy(new_U + z * xn * yn + offset, recvied_from_front_u_pos, DW * sizeof(float));
      // memcpy(new_V + z * xn * yn + offset, recvied_from_front_v_pos, DW * sizeof(float));
      // memcpy(new_W + z * xn * yn + offset, recvied_from_front_w_pos, DW * sizeof(float));
      recvied_from_back_u_pos += DW;
      recvied_from_back_v_pos += DW;
      recvied_from_back_w_pos += DW;
    }

    free(recvied_from_back_u);
    free(recvied_from_back_v);
    free(recvied_from_back_w);
  }
	// printf("Rank (%d, %d, %d) finish front\n", coords[0], coords[1], coords[2]);
  


  // 新增3个面的传输
}

void cover_ghost_points_top_front_right(int64_t *new_U, int64_t *new_V, int64_t *new_W,
                        int DL, int DH, int DW,int compress_xn, int compress_yn, int compress_zn, int front, int back, int top, int bot, int left, int right,
                        int *coords, MPI_Comm comm_cart,MPI_Comm mpi_comm, MPI_Status status)
{
	int rank = 0;
	MPI_Cart_rank(comm_cart, coords, &rank);
  // after all data is compressed, transmit the reset 3 surfaces
  // 这里需要传的是new_U, new_V, new_W对应的数据，而不是U, V, W！
  // front surface
  if(back != 1){
	// 发送back面block的数据到back面
	int64_t *back_surface_u = (int64_t *)malloc(sizeof(int64_t) * DL * DW);
	int64_t *back_surface_v = (int64_t *)malloc(sizeof(int64_t) * DL * DW);
	int64_t *back_surface_w = (int64_t *)malloc(sizeof(int64_t) * DL * DW);
	int64_t *back_surface_u_pos = back_surface_u;
	int64_t *back_surface_v_pos = back_surface_v;
	int64_t *back_surface_w_pos = back_surface_w;
	size_t offset_z = (bot ^ 1) * compress_xn * compress_yn; //z平面offset
	size_t offset_xy = (back ^ 1)*compress_xn + (left ^ 1); //xy平面offset
	// printf("BACK:  rank = %d,%d,%d, offset_z = %d, offset_xy = %d\n ", coords[0],coords[1],coords[2], offset_z, offset_xy);
	//得到这个new_U中的back面数据
	for(size_t i = 0; i < DL; i++){
		memcpy(back_surface_u_pos, new_U + offset_z + offset_xy + i * compress_xn * compress_yn, DW * sizeof(int64_t));
		memcpy(back_surface_v_pos, new_V + offset_z + offset_xy + i * compress_xn * compress_yn, DW * sizeof(int64_t));
		memcpy(back_surface_w_pos, new_W + offset_z + offset_xy + i * compress_xn * compress_yn, DW * sizeof(int64_t));
		back_surface_u_pos += DW;
		back_surface_v_pos += DW;
		back_surface_w_pos += DW;
  	}
	///

	int back_coords[3] = {coords[0], coords[1] - 1, coords[2]}; // 找到后面block的coord
	int back_rank;
	MPI_Cart_rank(comm_cart, back_coords, &back_rank);
	MPI_Send(back_surface_u, DL * DW, MPI_LONG, back_rank, 9, MPI_COMM_WORLD);
	MPI_Send(back_surface_v, DL * DW, MPI_LONG, back_rank, 10, MPI_COMM_WORLD);
	MPI_Send(back_surface_w, DL * DW, MPI_LONG, back_rank, 11, MPI_COMM_WORLD);
	free(back_surface_u);
	free(back_surface_v);
	free(back_surface_w);
  }
  if(front !=1){
	// 接收front面block的数据
	int64_t *recvied_from_front_u = (int64_t *)malloc(sizeof(int64_t) * DL * DW);
	int64_t *recvied_from_front_v = (int64_t *)malloc(sizeof(int64_t) * DL * DW);
	int64_t *recvied_from_front_w = (int64_t *)malloc(sizeof(int64_t) * DL * DW);
	int64_t *recvied_from_front_u_pos = recvied_from_front_u;
	int64_t *recvied_from_front_v_pos = recvied_from_front_v;
	int64_t *recvied_from_front_w_pos = recvied_from_front_w;
	int front_coords[3] = {coords[0], coords[1] + 1, coords[2]}; // 找到前面block的coord
	int front_rank;
	MPI_Cart_rank(comm_cart, front_coords, &front_rank);
	MPI_Recv(recvied_from_front_u, DL * DW, MPI_LONG, front_rank, 9, MPI_COMM_WORLD, &status);
	MPI_Recv(recvied_from_front_v, DL * DW, MPI_LONG, front_rank, 10, MPI_COMM_WORLD, &status);
	MPI_Recv(recvied_from_front_w, DL * DW, MPI_LONG, front_rank, 11, MPI_COMM_WORLD, &status);
	// 将接收到的数据放到new_U, new_V, new_W中
	size_t offset_z = (bot ^ 1) * compress_xn * compress_yn; //z平面offset
	size_t offset_xy = compress_xn * compress_yn - compress_xn + (left ^ 1); //xy平面offset
	// printf("FRONT rank = %d,%d,%d, offset_z = %d, offset_xy = %d\n", coords[0],coords[1],coords[2], offset_z, offset_xy);
	for(size_t i = 0; i < DL; i++){

		memcpy(new_U + offset_z + offset_xy + i * compress_xn * compress_yn, recvied_from_front_u_pos, DW * sizeof(int64_t));
		memcpy(new_V + offset_z + offset_xy + i * compress_xn * compress_yn, recvied_from_front_v_pos, DW * sizeof(int64_t));
		memcpy(new_W + offset_z + offset_xy + i * compress_xn * compress_yn, recvied_from_front_w_pos, DW * sizeof(int64_t));
		recvied_from_front_u_pos += DW;
		recvied_from_front_v_pos += DW;
		recvied_from_front_w_pos += DW;
	}
	if(rank == 1){
		// printf("%f %f %f %f %f\n", recvied_from_front_u[255*256], recvied_from_front_u[255*256 + 1], recvied_from_front_u[255*256 + 2], recvied_from_front_u[255*256 + 3], recvied_from_front_u[255*256 + 4]);
	}
	free(recvied_from_front_u);
	free(recvied_from_front_v);
	free(recvied_from_front_w);
  }
	// printf("Rank (%d, %d, %d) finish front\n", coords[0], coords[1], coords[2]);
  // right surface
  if(left != 1){
	// 发送left面block的数据到left面
	int64_t *left_surface_u = (int64_t *)malloc(sizeof(int64_t) * DL * DH);
	int64_t *left_surface_v = (int64_t *)malloc(sizeof(int64_t) * DL * DH);
	int64_t *left_surface_w = (int64_t *)malloc(sizeof(int64_t) * DL * DH);
	int64_t *left_surface_u_pos = left_surface_u;
	int64_t *left_surface_v_pos = left_surface_v;
	int64_t *left_surface_w_pos = left_surface_w;
	//得到这个new_U中的left面数据
	for(size_t i = 0; i < DL; i++){
		for (size_t j = 0; j < DH; j++)
			{
			size_t offset_z = (bot ^ 1) * compress_xn * compress_yn; //z平面offset
			size_t offset_xy = (back ^ 1) * compress_xn + (left ^ 1); //xy平面offset
			*left_surface_u_pos = new_U[offset_z + offset_xy + i * compress_xn * compress_yn + j * compress_xn];
			*left_surface_v_pos = new_V[offset_z + offset_xy + i * compress_xn * compress_yn + j * compress_xn];
			*left_surface_w_pos = new_W[offset_z + offset_xy + i * compress_xn * compress_yn + j * compress_xn];
			left_surface_u_pos++;
			left_surface_v_pos++;
			left_surface_w_pos++;
		}
		
  	}
	int left_coords[3] = {coords[0], coords[1], coords[2]-1}; // 找到左面block的coord
	int left_rank;
	MPI_Cart_rank(comm_cart, left_coords, &left_rank);
	MPI_Send(left_surface_u, DL * DH, MPI_LONG, left_rank, 12, MPI_COMM_WORLD);
	MPI_Send(left_surface_v, DL * DH, MPI_LONG, left_rank, 13, MPI_COMM_WORLD);
	MPI_Send(left_surface_w, DL * DH, MPI_LONG, left_rank, 14, MPI_COMM_WORLD);
	free(left_surface_u);
	free(left_surface_v);
	free(left_surface_w);
  }
  if(right != 1){
	// 接收right面block的数据
	int64_t *recvied_from_right_u = (int64_t *)malloc(sizeof(int64_t) * DL * DH);
	int64_t *recvied_from_right_v = (int64_t *)malloc(sizeof(int64_t) * DL * DH);
	int64_t *recvied_from_right_w = (int64_t *)malloc(sizeof(int64_t) * DL * DH);
	int64_t *recvied_from_right_u_pos = recvied_from_right_u;
	int64_t *recvied_from_right_v_pos = recvied_from_right_v;
	int64_t *recvied_from_right_w_pos = recvied_from_right_w;
	int right_coords[3] = {coords[0], coords[1], coords[2]+1}; // 找到右面block的coord
	int right_rank;
	MPI_Cart_rank(comm_cart, right_coords, &right_rank);
	MPI_Recv(recvied_from_right_u, DL * DH, MPI_LONG, right_rank, 12, MPI_COMM_WORLD, &status);
	MPI_Recv(recvied_from_right_v, DL * DH, MPI_LONG, right_rank, 13, MPI_COMM_WORLD, &status);
	MPI_Recv(recvied_from_right_w, DL * DH, MPI_LONG, right_rank, 14, MPI_COMM_WORLD, &status);
	// 将接收到的数据放到new_U中
	for(size_t i = 0; i < DL; i++){
		for (size_t j = 0; j < DH; j++)
		{
			size_t offset_z = (bot ^ 1) * compress_xn * compress_yn; //z平面offset
			size_t offset_xy = (back ^ 1) * compress_xn +  compress_xn -1; //xy平面offset
			new_U[offset_z + offset_xy + i * compress_xn * compress_yn + j * compress_xn] = *recvied_from_right_u_pos;
			new_V[offset_z + offset_xy + i * compress_xn * compress_yn + j * compress_xn] = *recvied_from_right_v_pos;
			new_W[offset_z + offset_xy + i * compress_xn * compress_yn + j * compress_xn] = *recvied_from_right_w_pos;
			recvied_from_right_u_pos ++;
			recvied_from_right_v_pos ++;
			recvied_from_right_w_pos ++;
		}
	}
	free(recvied_from_right_u);
	free(recvied_from_right_v);
	free(recvied_from_right_w);
  }
	// printf("Rank (%d, %d, %d) finish right\n", coords[0], coords[1], coords[2]);


  // top surface
  if(bot != 1){
	// 发送bot面block的数据
	int64_t *bot_surface_u = (int64_t *)malloc(sizeof(int64_t) * DW * DH);
	int64_t *bot_surface_v = (int64_t *)malloc(sizeof(int64_t) * DW * DH);
	int64_t *bot_surface_w = (int64_t *)malloc(sizeof(int64_t) * DW * DH);
	int64_t *bot_surface_u_pos = bot_surface_u;
	int64_t *bot_surface_v_pos = bot_surface_v;
	int64_t *bot_surface_w_pos = bot_surface_w;
	for(size_t i = 0; i < DH; i++){
		size_t offset_z = (bot ^ 1) * compress_xn * compress_yn; //z平面offset
		size_t offset_xy = (back ^ 1) * compress_xn +(left ^ 1); //xy平面offset
		memcpy(bot_surface_u_pos, new_U + offset_z + offset_xy + i * compress_xn, DW* sizeof(int64_t));
		memcpy(bot_surface_v_pos, new_V + offset_z + offset_xy + i * compress_xn, DW* sizeof(int64_t));
		memcpy(bot_surface_w_pos, new_W + offset_z + offset_xy + i * compress_xn, DW* sizeof(int64_t));
		bot_surface_u_pos += DW;
		bot_surface_v_pos += DW;
		bot_surface_w_pos += DW;
	}
	int bot_coords[3] = {coords[0]-1, coords[1], coords[2]}; // 找到bot面block的coord
	int bot_rank;
	MPI_Cart_rank(comm_cart, bot_coords, &bot_rank);
	MPI_Send(bot_surface_u, DW * DH, MPI_LONG, bot_rank, 15, MPI_COMM_WORLD);
	MPI_Send(bot_surface_v, DW * DH, MPI_LONG, bot_rank, 16, MPI_COMM_WORLD);
	MPI_Send(bot_surface_w, DW * DH, MPI_LONG, bot_rank, 17, MPI_COMM_WORLD);
	free(bot_surface_u);
	free(bot_surface_v);
	free(bot_surface_w);
  }
  if(top != 1){
	// 接收top面block的数据
	int64_t *recvied_from_top_u = (int64_t *)malloc(sizeof(int64_t) * DW * DH);
	int64_t *recvied_from_top_v = (int64_t *)malloc(sizeof(int64_t) * DW * DH);
	int64_t *recvied_from_top_w = (int64_t *)malloc(sizeof(int64_t) * DW * DH);
	int64_t *recvied_from_top_u_pos = recvied_from_top_u;
	int64_t *recvied_from_top_v_pos = recvied_from_top_v;
	int64_t *recvied_from_top_w_pos = recvied_from_top_w;
	int top_coords[3] = {coords[0]+1, coords[1], coords[2]}; // 找到top面block的coord
	int top_rank;
	MPI_Cart_rank(comm_cart, top_coords, &top_rank);
	MPI_Recv(recvied_from_top_u, DW * DH, MPI_LONG, top_rank, 15, MPI_COMM_WORLD, &status);
	MPI_Recv(recvied_from_top_v, DW * DH, MPI_LONG, top_rank, 16, MPI_COMM_WORLD, &status);
	MPI_Recv(recvied_from_top_w, DW * DH, MPI_LONG, top_rank, 17, MPI_COMM_WORLD, &status);
	// 将接收到的数据放到new_U中
	for(size_t i = 0; i < DH; i++){
		size_t offset_z = compress_xn * compress_yn * (compress_zn-1); //z平面offset
		size_t offset_xy = (back ^ 1) * compress_xn +  (left ^ 1); //xy平面offset
		memcpy(new_U + offset_z + offset_xy + i * compress_xn, recvied_from_top_u_pos, DW* sizeof(int64_t));
		memcpy(new_V + offset_z + offset_xy + i * compress_xn, recvied_from_top_v_pos, DW* sizeof(int64_t));
		memcpy(new_W + offset_z + offset_xy + i * compress_xn, recvied_from_top_w_pos, DW* sizeof(int64_t));
		recvied_from_top_u_pos += DW;
		recvied_from_top_v_pos += DW;
		recvied_from_top_w_pos += DW;
	}
	free(recvied_from_top_u);
	free(recvied_from_top_v);
	free(recvied_from_top_w);
  }
	// printf("Rank (%d, %d, %d) finish top\n", coords[0], coords[1], coords[2]);

}


template<typename T_data>
unsigned char *
sz_compress_cp_preserve_sos_3d_online_fp_parallel(const T_data * U, const T_data * V, const T_data * W, size_t r1, size_t r2, size_t r3, size_t& compressed_size, bool transpose, double max_pwr_eb){
	// std::cout << "sz_compress_cp_preserve_sos_3d_online_fp" << std::endl;
	
	MPI_Status status;
	int rank, size;
	int periods[3] = {0, 0, 0};
	int dims[3] = {0};
	int coords[3] = {0};
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
	int size_cubert = std::round(std::pow(size, 1.0 / 3));
 	dims[0] = dims[1] = dims[2] = size_cubert; // 边长
  	// printf("size-cubert = %d\n",size_cubert);
	// if(rank == 0){
	// 	printf("rank = %d, size = %d, size_cubert = %d\n", rank, size, size_cubert);
	// }
  	MPI_Comm comm_cart;
  	MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &comm_cart);
	// printf("create communicator done, dims: %d %d %d\n", dims[0], dims[1], dims[2]);
  	MPI_Cart_coords(comm_cart, rank, 3, coords);
	// printf("create MPI_Cart_coords done, coords: %d %d %d\n", coords[0], coords[1], coords[2]);
	int compress_xn = r3;
	int compress_yn = r2;
	int compress_zn = r1;
	int front = 0, back = 0, top = 0, bot = 0, left = 0, right = 0;

	using T = int64_t;
	size_t num_elements = r1 * r2 * r3;
  if (size == 1){
    bot = 1;
    top = 1;
    left = 1;
    right = 1;
    front = 1;
    back = 1;
    compress_xn = r3;
    compress_yn = r2;
    compress_zn = r1;

  }
  else{
  if (coords[0] == 0)
  {
    bot = 1;
    compress_zn += 1;
  }
  else if (coords[0] == size_cubert - 1)
  {
    top = 1;
    compress_zn += 1;
  }
  else{
    compress_zn += 2;}

  if (coords[1] == 0)
  {
    back = 1;

    compress_yn += 1;
  }
  else if (coords[1] == size_cubert - 1)
  {
    front = 1;
    compress_yn += 1;
  }
  else{
    compress_yn += 2;}

  if (coords[2] == 0)
  {
    left = 1;
    compress_xn += 1;
  }
  else if (coords[2] == size_cubert - 1)
  {
    right = 1;
    compress_xn += 1;
  }
  else{
    compress_xn += 2;}

  }


	size_t compress_block_size = compress_xn * compress_yn * compress_zn;
	T_data *new_U = (T_data *)malloc(sizeof(T_data) * compress_block_size);
  	T_data *new_V = (T_data *)malloc(sizeof(T_data) * compress_block_size);
  	T_data *new_W = (T_data *)malloc(sizeof(T_data) * compress_block_size);
	memset(new_U, 0, sizeof(T_data) * compress_block_size);
  	memset(new_V, 0, sizeof(T_data) * compress_block_size);
  	memset(new_W, 0, sizeof(T_data) * compress_block_size);

    // first round communication
	cover_ghost_points_bot_back_left(U,V,W,new_U,new_V,new_W,r1,r2,r3,compress_xn,compress_yn,compress_zn,front,back,top,bot,left,right,coords,comm_cart,MPI_COMM_WORLD,status);

	T * U_fp = (T *) malloc(compress_block_size*sizeof(T));
	T * V_fp = (T *) malloc(compress_block_size*sizeof(T));
	T * W_fp = (T *) malloc(compress_block_size*sizeof(T));
	T range = 0;
	T vector_field_scaling_factor = convert_to_fixed_point(new_U, new_V, new_W, compress_block_size, U_fp, V_fp, W_fp, range);
	// printf("fixed point range = %lld\n", range);
	int * eb_quant_index = (int *) malloc(num_elements*sizeof(int));
	int * data_quant_index = (int *) malloc(3*num_elements*sizeof(int));
	int * eb_quant_index_pos = eb_quant_index;
	int * data_quant_index_pos = data_quant_index;
	// next, row by row
	const int base = 2;
	const double log_of_base = log2(base);
	const int capacity = 65536;
	const int intv_radius = (capacity >> 1);
	T max_eb = range * max_pwr_eb;
	unpred_vec<T_data> unpred_data;
	// ptrdiff_t dim0_offset = r2 * r3;
	// ptrdiff_t dim1_offset = r3;
	ptrdiff_t dim0_offset_ext = compress_yn * compress_xn;
	ptrdiff_t dim1_offset_ext = compress_xn;
	ptrdiff_t cell_dim0_offset_ext = (compress_yn-1) * (compress_xn-1);
	ptrdiff_t cell_dim1_offset_ext = compress_xn-1;
	int simplex_offset[24];
	int index_offset[24][3][3];
	int offset[24][3];
	compute_offset(dim0_offset_ext, dim1_offset_ext, cell_dim0_offset_ext, cell_dim1_offset_ext, simplex_offset, index_offset, offset);
	T threshold = 1;
	// check cp for all cells
	// std::cout << "start cp checking\n";

	//write new_U to file
	// IO::posix_write(("u_rank" + std::to_string(rank) + "_first_round.out").c_str(), new_U, compress_block_size);
	// IO::posix_write(("v_rank" + std::to_string(rank) + "_first_round.out").c_str(), new_V, compress_block_size);
	// IO::posix_write(("w_rank" + std::to_string(rank) + "_first_round.out").c_str(), new_W, compress_block_size);
	

	// printf("compress_xn: %d, compress_yn: %d, compress_zn: %d, compress_block_size: %d\n", compress_xn, compress_yn, compress_zn, compress_block_size);

	vector<bool> cp_exist = compute_cp(U_fp, V_fp, W_fp, compress_zn, compress_yn, compress_xn);
	// bool top = 0, bottom = 0, left = 0, right = 0, front = 0, back = 0;
	// compress (r1-1)(r2-1)(r3-1) cube in (r1+2)(r2+2)(r3+2) cube
	ptrdiff_t start_offset = (1-bot)*dim0_offset_ext + (1-back)*dim1_offset_ext + (1-left);
	T * start_U_pos = U_fp + start_offset;
	T * start_V_pos = V_fp + start_offset;
	T * start_W_pos = W_fp + start_offset;
	T * cur_U_pos = NULL;
	T * cur_V_pos = NULL;
	T * cur_W_pos = NULL;
	// std::cout << "start compression\n";
	// compress corner cube (r1-1)*(r2-1)*(r3-1)
	// std::cout << "compress cube\n";
	for(int i=0; i<r1-1; i++){
		for(int j=0; j<r2-1; j++){
			cur_U_pos = start_U_pos + i*dim0_offset_ext + j*dim1_offset_ext;
			cur_V_pos = start_V_pos + i*dim0_offset_ext + j*dim1_offset_ext;
			cur_W_pos = start_W_pos + i*dim0_offset_ext + j*dim1_offset_ext;
			for(int k=0; k<r3-1; k++){
				T required_eb = max_eb;
				if(((i==0) && (j==0)) || ((i==0) && (k==0)) || ((j==0) && (k==0))){
					// set eb for edges to 0
					required_eb = 0;
				}
				// if((i==0) || (j==0) || (k==0)){
				// 	// set eb for edges to 0
				// 	required_eb = 0;
				// }
				else{
					int i_ = i + (1 - bot);
					int j_ = j + (1 - back);
					int k_ = k + (1 - left);
					// derive eb given 24 adjacent simplex
					for(int n=0; n<24; n++){
						bool in_mesh = true;
						for(int p=0; p<3; p++){
							// reversed order!
							if(!(in_range(i_ + index_offset[n][p][2], (int)compress_zn) && in_range(j_ + index_offset[n][p][1], (int)compress_yn) && in_range(k_ + index_offset[n][p][0], (int)compress_xn))){
								in_mesh = false;
								break;
							}
						}
						if(in_mesh){
							int index = simplex_offset[n] + 6*(i_*cell_dim0_offset_ext + j_*cell_dim1_offset_ext + k_);
							if(cp_exist[index]){
								required_eb = 0;
								break;
							}
							required_eb = MINF(required_eb, derive_cp_abs_eb_sos_online(
								cur_U_pos[offset[n][0]], cur_U_pos[offset[n][1]], cur_U_pos[offset[n][2]], *cur_U_pos,
								cur_V_pos[offset[n][0]], cur_V_pos[offset[n][1]], cur_V_pos[offset[n][2]], *cur_V_pos,
								cur_W_pos[offset[n][0]], cur_W_pos[offset[n][1]], cur_W_pos[offset[n][2]], *cur_W_pos));
						}
					}			
				}
				T abs_eb = required_eb;
				*eb_quant_index_pos = eb_exponential_quantize(abs_eb, base, log_of_base, threshold);
				if(abs_eb > 0){
					bool unpred_flag = false;
					T decompressed[3];
					// compress vector fields
					T * data_pos[3] = {cur_U_pos, cur_V_pos, cur_W_pos};
					for(int p=0; p<3; p++){
						T * cur_data_pos = data_pos[p];
						T cur_data = *cur_data_pos;
						// get adjacent data and perform Lorenzo
						/*
							d6	X
							d4	d5
							d2	d3
							d0	d1
						*/
						T d0 = (i && j && k) ? cur_data_pos[- dim0_offset_ext - dim1_offset_ext - 1] : 0;
						T d1 = (i && j) ? cur_data_pos[- dim0_offset_ext - dim1_offset_ext] : 0;
						T d2 = (i && k) ? cur_data_pos[- dim0_offset_ext - 1] : 0;
						T d3 = (i) ? cur_data_pos[- dim0_offset_ext] : 0;
						T d4 = (j && k) ? cur_data_pos[- dim1_offset_ext - 1] : 0;
						T d5 = (j) ? cur_data_pos[- dim1_offset_ext] : 0;
						T d6 = (k) ? cur_data_pos[- 1] : 0;
						T pred = d0 + d3 + d5 + d6 - d1 - d2 - d4;
						T diff = cur_data - pred;
						T quant_diff = std::abs(diff) / abs_eb + 1;
						if(quant_diff < capacity){
							quant_diff = (diff > 0) ? quant_diff : -quant_diff;
							int quant_index = (int)(quant_diff/2) + intv_radius;
							data_quant_index_pos[p] = quant_index;
							decompressed[p] = pred + 2 * (quant_index - intv_radius) * abs_eb; 
							// check original data
							if(std::abs(decompressed[p] - cur_data) >= required_eb){
								unpred_flag = true;
								break;
							}
						}
						else{
							unpred_flag = true;
							break;
						}
					}
					if(unpred_flag){
						*(eb_quant_index_pos ++) = 0;
						ptrdiff_t offset = cur_U_pos - U_fp;
						unpred_data.push_back(new_U[offset]);
						unpred_data.push_back(new_V[offset]);
						unpred_data.push_back(new_W[offset]);
					}
					else{
						eb_quant_index_pos ++;
						data_quant_index_pos += 3;
						*cur_U_pos = decompressed[0];
						*cur_V_pos = decompressed[1];
						*cur_W_pos = decompressed[2];
					}
				}
				else{
					// record as unpredictable data
					*(eb_quant_index_pos ++) = 0;
					ptrdiff_t offset = cur_U_pos - U_fp;
					unpred_data.push_back(new_U[offset]);
					unpred_data.push_back(new_V[offset]);
					unpred_data.push_back(new_W[offset]);
				}
				cur_U_pos ++, cur_V_pos ++, cur_W_pos ++;
			}
			// skip the last element
		}
		// skip the last line
	}

	// 2nd-round communicate
	cover_ghost_points_top_front_right(U_fp,V_fp,W_fp,r1,r2,r3,compress_xn,compress_yn,compress_zn,front,back,top,bot,left,right,coords,comm_cart,MPI_COMM_WORLD,status);
	// write 8 files new_u
	// IO::posix_write(("u_rank" + std::to_string(rank) + "_second_round.out").c_str(), new_U, compress_block_size);
	// IO::posix_write(("v_rank" + std::to_string(rank) + "_second_round.out").c_str(), new_V, compress_block_size);
	// IO::posix_write(("w_rank" + std::to_string(rank) + "_second_round.out").c_str(), new_W, compress_block_size);
	// compress top surface (r2-1)*(r3-1)
	{
		// std::cout << "compress top\n";
		int i = r1 - 1;
		for(int j=0; j<r2-1; j++){
			cur_U_pos = start_U_pos + i*dim0_offset_ext + j*dim1_offset_ext, cur_V_pos = start_V_pos + i*dim0_offset_ext + j*dim1_offset_ext, cur_W_pos = start_W_pos + i*dim0_offset_ext + j*dim1_offset_ext;
			for(int k=0; k<r3-1; k++){
				T required_eb = max_eb;
				if((j==0) || (k==0)){
				// if(true){
					// set eb for edges to 0
					required_eb = 0;
				}
				else{
					int i_ = i + (1 - bot);
					int j_ = j + (1 - back);
					int k_ = k + (1 - left);
					// derive eb given 24 adjacent simplex
					for(int n=0; n<24; n++){
						bool in_mesh = true;
						for(int p=0; p<3; p++){
							// reversed order!
							if(!(in_range(i_ + index_offset[n][p][2], (int)compress_zn) && in_range(j_ + index_offset[n][p][1], (int)compress_yn) && in_range(k_ + index_offset[n][p][0], (int)compress_xn))){
								in_mesh = false;
								break;
							}
						}
						if(in_mesh){
							// int index = simplex_offset[n] + 6*(i_*cell_dim0_offset_ext + j_*cell_dim1_offset_ext + k_);
							// if(cp_exist[index]){
							// 	required_eb = 0;
							// 	break;
							// }
							int indices[4];
							int vertex_index = i_*dim0_offset_ext + j_*dim1_offset_ext + k_;
							for(int p=0; p<3; p++){
								indices[p] = vertex_index + offset[n][p];
							}
							indices[3] = vertex_index;
							T vf[4][3];
							for(int p=0; p<4; p++){
								vf[p][0] = U_fp[indices[p]];
								vf[p][1] = V_fp[indices[p]];
								vf[p][2] = W_fp[indices[p]];
							}
							if(check_cp(vf, indices) == 1){
								required_eb = 0;
								break;
							}
							required_eb = MINF(required_eb, derive_cp_abs_eb_sos_online(
								cur_U_pos[offset[n][0]], cur_U_pos[offset[n][1]], cur_U_pos[offset[n][2]], *cur_U_pos,
								cur_V_pos[offset[n][0]], cur_V_pos[offset[n][1]], cur_V_pos[offset[n][2]], *cur_V_pos,
								cur_W_pos[offset[n][0]], cur_W_pos[offset[n][1]], cur_W_pos[offset[n][2]], *cur_W_pos));
						}
					}			
				}
				T abs_eb = required_eb;
				*eb_quant_index_pos = eb_exponential_quantize(abs_eb, base, log_of_base, threshold);
				if(abs_eb > 0){
					bool unpred_flag = false;
					T decompressed[3];
					// compress vector fields
					T * data_pos[3] = {cur_U_pos, cur_V_pos, cur_W_pos};
					for(int p=0; p<3; p++){
						T * cur_data_pos = data_pos[p];
						T cur_data = *cur_data_pos;
						// get adjacent data and perform Lorenzo
						/*
							d6	X
							d4	d5
							d2	d3
							d0	d1
						*/
						T d0 = cur_data_pos[- dim0_offset_ext - dim1_offset_ext - 1];
						T d1 = cur_data_pos[- dim0_offset_ext - dim1_offset_ext];
						T d2 = cur_data_pos[- dim0_offset_ext - 1];
						T d3 = cur_data_pos[- dim0_offset_ext];
						T d4 = cur_data_pos[- dim1_offset_ext - 1];
						T d5 = cur_data_pos[- dim1_offset_ext];
						T d6 = cur_data_pos[- 1];
						T pred = d0 + d3 + d5 + d6 - d1 - d2 - d4;
						T diff = cur_data - pred;
						T quant_diff = std::abs(diff) / abs_eb + 1;
						if(quant_diff < capacity){
							quant_diff = (diff > 0) ? quant_diff : -quant_diff;
							int quant_index = (int)(quant_diff/2) + intv_radius;
							data_quant_index_pos[p] = quant_index;
							decompressed[p] = pred + 2 * (quant_index - intv_radius) * abs_eb; 
							// check original data
							if(std::abs(decompressed[p] - cur_data) >= required_eb){
								unpred_flag = true;
								break;
							}
						}
						else{
							unpred_flag = true;
							break;
						}
					}
					if(unpred_flag){
						*(eb_quant_index_pos ++) = 0;
						ptrdiff_t offset = cur_U_pos - U_fp;
						unpred_data.push_back(new_U[offset]);
						unpred_data.push_back(new_V[offset]);
						unpred_data.push_back(new_W[offset]);
					}
					else{
						eb_quant_index_pos ++;
						data_quant_index_pos += 3;
						*cur_U_pos = decompressed[0];
						*cur_V_pos = decompressed[1];
						*cur_W_pos = decompressed[2];
					}
				}
				else{
					// record as unpredictable data
					*(eb_quant_index_pos ++) = 0;
					ptrdiff_t offset = cur_U_pos - U_fp;
					unpred_data.push_back(new_U[offset]);
					unpred_data.push_back(new_V[offset]);
					unpred_data.push_back(new_W[offset]);
				}
				cur_U_pos ++, cur_V_pos ++, cur_W_pos ++;
			}
		}
	}
	// compress right surface r1*(r2-1)
	{
		// printf("Rank %d compress right\n", rank);fflush(stdout);
		int k = r3-1;
		for(int i=0; i<r1; i++){
			cur_U_pos = start_U_pos + i*dim0_offset_ext + k, cur_V_pos = start_V_pos + i*dim0_offset_ext + k, cur_W_pos = start_W_pos + i*dim0_offset_ext + k;
			for(int j=0; j<r2-1; j++){
				T required_eb = max_eb;
				if((i == 0) || (i == r1-1) || (j == 0)){
				// if(true){
					// set eb for edges to 0
					required_eb = 0;
				}
				else{
					int i_ = i + (1 - bot);
					int j_ = j + (1 - back);
					int k_ = k + (1 - left);
					// derive eb given 24 adjacent simplex
					for(int n=0; n<24; n++){
						bool in_mesh = true;
						for(int p=0; p<3; p++){
							// reversed order!
							if(!(in_range(i_ + index_offset[n][p][2], (int)compress_zn) && in_range(j_ + index_offset[n][p][1], (int)compress_yn) && in_range(k_ + index_offset[n][p][0], (int)compress_xn))){
								in_mesh = false;
								break;
							}
						}
						if(in_mesh){
							// int index = simplex_offset[n] + 6*(i_*cell_dim0_offset_ext + j_*cell_dim1_offset_ext + k_);
							// if(cp_exist[index]){
							// 	required_eb = 0;
							// 	break;
							// }
							int indices[4];
							int vertex_index = i_*dim0_offset_ext + j_*dim1_offset_ext + k_;
							for(int p=0; p<3; p++){
								indices[p] = vertex_index + offset[n][p];
							}
							indices[3] = vertex_index;
							T vf[4][3];
							for(int p=0; p<4; p++){
								vf[p][0] = U_fp[indices[p]];
								vf[p][1] = V_fp[indices[p]];
								vf[p][2] = W_fp[indices[p]];
							}
							if(check_cp(vf, indices) == 1){
								required_eb = 0;
								break;
							}
							required_eb = MINF(required_eb, derive_cp_abs_eb_sos_online(
								cur_U_pos[offset[n][0]], cur_U_pos[offset[n][1]], cur_U_pos[offset[n][2]], *cur_U_pos,
								cur_V_pos[offset[n][0]], cur_V_pos[offset[n][1]], cur_V_pos[offset[n][2]], *cur_V_pos,
								cur_W_pos[offset[n][0]], cur_W_pos[offset[n][1]], cur_W_pos[offset[n][2]], *cur_W_pos));
						}
					}			
				}
				T abs_eb = required_eb;
				*eb_quant_index_pos = eb_exponential_quantize(abs_eb, base, log_of_base, threshold);
				if(abs_eb > 0){
					bool unpred_flag = false;
					T decompressed[3];
					// compress vector fields
					T * data_pos[3] = {cur_U_pos, cur_V_pos, cur_W_pos};
					for(int p=0; p<3; p++){
						T * cur_data_pos = data_pos[p];
						T cur_data = *cur_data_pos;
						// get adjacent data and perform Lorenzo
						/*
							d6	X
							d4	d5
							d2	d3
							d0	d1
						*/
						T d0 = cur_data_pos[- dim0_offset_ext - dim1_offset_ext - 1];
						T d1 = cur_data_pos[- dim0_offset_ext - dim1_offset_ext];
						T d2 = cur_data_pos[- dim0_offset_ext - 1];
						T d3 = cur_data_pos[- dim0_offset_ext];
						T d4 = cur_data_pos[- dim1_offset_ext - 1];
						T d5 = cur_data_pos[- dim1_offset_ext];
						T d6 = cur_data_pos[- 1];
						T pred = d0 + d3 + d5 + d6 - d1 - d2 - d4;
						T diff = cur_data - pred;
						T quant_diff = std::abs(diff) / abs_eb + 1;
						if(quant_diff < capacity){
							quant_diff = (diff > 0) ? quant_diff : -quant_diff;
							int quant_index = (int)(quant_diff/2) + intv_radius;
							data_quant_index_pos[p] = quant_index;
							decompressed[p] = pred + 2 * (quant_index - intv_radius) * abs_eb; 
							// check original data
							if(std::abs(decompressed[p] - cur_data) >= required_eb){
								unpred_flag = true;
								break;
							}
						}
						else{
							unpred_flag = true;
							break;
						}
					}
					if(unpred_flag){
						*(eb_quant_index_pos ++) = 0;
						ptrdiff_t offset = cur_U_pos - U_fp;
						unpred_data.push_back(new_U[offset]);
						unpred_data.push_back(new_V[offset]);
						unpred_data.push_back(new_W[offset]);
					}
					else{
						eb_quant_index_pos ++;
						data_quant_index_pos += 3;
						*cur_U_pos = decompressed[0];
						*cur_V_pos = decompressed[1];
						*cur_W_pos = decompressed[2];
					}
				}
				else{
					// record as unpredictable data
					*(eb_quant_index_pos ++) = 0;
					ptrdiff_t offset = cur_U_pos - U_fp;
					unpred_data.push_back(new_U[offset]);
					unpred_data.push_back(new_V[offset]);
					unpred_data.push_back(new_W[offset]);
				}
				cur_U_pos += dim1_offset_ext, cur_V_pos += dim1_offset_ext, cur_W_pos += dim1_offset_ext;
			}
		}		
	}
	// compress back surface r1*r3
	{
		// printf("Rank %d compress back\n", rank);fflush(stdout);
		int j = r2-1;
		for(int i=0; i<r1; i++){
			cur_U_pos = start_U_pos + i*dim0_offset_ext + j*dim1_offset_ext, cur_V_pos = start_V_pos + i*dim0_offset_ext + j*dim1_offset_ext, cur_W_pos = start_W_pos + i*dim0_offset_ext + j*dim1_offset_ext;
			for(int k=0; k<r3; k++){
				T required_eb = max_eb;
				if((i == 0) || (i == r1-1) || (k == 0) || (k==r3-1)){
				// if(true){
					// set eb for edges to 0
					required_eb = 0;
				}
				else{
					int i_ = i + (1 - bot);
					int j_ = j + (1 - back);
					int k_ = k + (1 - left);
					// derive eb given 24 adjacent simplex
					for(int n=0; n<24; n++){
						bool in_mesh = true;
						for(int p=0; p<3; p++){
							// reversed order!
							if(!(in_range(i_ + index_offset[n][p][2], (int)compress_zn) && in_range(j_ + index_offset[n][p][1], (int)compress_yn) && in_range(k_ + index_offset[n][p][0], (int)compress_xn))){
								in_mesh = false;
								break;
							}
						}
						if(in_mesh){
							// int index = simplex_offset[n] + 6*(i_*cell_dim0_offset_ext + j_*cell_dim1_offset_ext + k_);
							// if(cp_exist[index]){
							// 	required_eb = 0;
							// 	break;
							// }
							int indices[4];
							int vertex_index = i_*dim0_offset_ext + j_*dim1_offset_ext + k_;
							for(int p=0; p<3; p++){
								indices[p] = vertex_index + offset[n][p];
							}
							indices[3] = vertex_index;
							T vf[4][3];
							for(int p=0; p<4; p++){
								vf[p][0] = U_fp[indices[p]];
								vf[p][1] = V_fp[indices[p]];
								vf[p][2] = W_fp[indices[p]];
							}
							if(check_cp(vf, indices) == 1){
								required_eb = 0;
								break;
							}
							required_eb = MINF(required_eb, derive_cp_abs_eb_sos_online(
								cur_U_pos[offset[n][0]], cur_U_pos[offset[n][1]], cur_U_pos[offset[n][2]], *cur_U_pos,
								cur_V_pos[offset[n][0]], cur_V_pos[offset[n][1]], cur_V_pos[offset[n][2]], *cur_V_pos,
								cur_W_pos[offset[n][0]], cur_W_pos[offset[n][1]], cur_W_pos[offset[n][2]], *cur_W_pos));
						}
					}			
				}
				T abs_eb = required_eb;
				*eb_quant_index_pos = eb_exponential_quantize(abs_eb, base, log_of_base, threshold);
				if(abs_eb > 0){
					bool unpred_flag = false;
					T decompressed[3];
					// compress vector fields
					T * data_pos[3] = {cur_U_pos, cur_V_pos, cur_W_pos};
					for(int p=0; p<3; p++){
						T * cur_data_pos = data_pos[p];
						T cur_data = *cur_data_pos;
						// get adjacent data and perform Lorenzo
						/*
							d6	X
							d4	d5
							d2	d3
							d0	d1
						*/
						T d0 = cur_data_pos[- dim0_offset_ext - dim1_offset_ext - 1];
						T d1 = cur_data_pos[- dim0_offset_ext - dim1_offset_ext];
						T d2 = cur_data_pos[- dim0_offset_ext - 1];
						T d3 = cur_data_pos[- dim0_offset_ext];
						T d4 = cur_data_pos[- dim1_offset_ext - 1];
						T d5 = cur_data_pos[- dim1_offset_ext];
						T d6 = cur_data_pos[- 1];
						T pred = d0 + d3 + d5 + d6 - d1 - d2 - d4;
						T diff = cur_data - pred;
						T quant_diff = std::abs(diff) / abs_eb + 1;
						if(quant_diff < capacity){
							quant_diff = (diff > 0) ? quant_diff : -quant_diff;
							int quant_index = (int)(quant_diff/2) + intv_radius;
							data_quant_index_pos[p] = quant_index;
							decompressed[p] = pred + 2 * (quant_index - intv_radius) * abs_eb; 
							// check original data
							if(std::abs(decompressed[p] - cur_data) >= required_eb){
								unpred_flag = true;
								break;
							}
						}
						else{
							unpred_flag = true;
							break;
						}
					}
					if(unpred_flag){
						*(eb_quant_index_pos ++) = 0;
						ptrdiff_t offset = cur_U_pos - U_fp;
						unpred_data.push_back(new_U[offset]);
						unpred_data.push_back(new_V[offset]);
						unpred_data.push_back(new_W[offset]);
					}
					else{
						eb_quant_index_pos ++;
						data_quant_index_pos += 3;
						*cur_U_pos = decompressed[0];
						*cur_V_pos = decompressed[1];
						*cur_W_pos = decompressed[2];
					}
				}
				else{
					// record as unpredictable data
					*(eb_quant_index_pos ++) = 0;
					ptrdiff_t offset = cur_U_pos - U_fp;
					unpred_data.push_back(new_U[offset]);
					unpred_data.push_back(new_V[offset]);
					unpred_data.push_back(new_W[offset]);
				}
				cur_U_pos ++, cur_V_pos ++, cur_W_pos ++;
			}
		}		
	}
	free(U_fp);
	free(V_fp);
	free(W_fp);
	free(new_U);
	free(new_V);
	free(new_W);
	// printf("Rank %d finish data processing\n", rank);fflush(stdout);
	// printf("offset eb_q, data_q, unpred: %ld %ld %ld\n", eb_quant_index_pos - eb_quant_index, data_quant_index_pos - data_quant_index, unpred_data.size());
	unsigned char * compressed = (unsigned char *) malloc(3*num_elements*sizeof(T));
	unsigned char * compressed_pos = compressed;
	write_variable_to_dst(compressed_pos, vector_field_scaling_factor);
	write_variable_to_dst(compressed_pos, base);
	write_variable_to_dst(compressed_pos, threshold);
	write_variable_to_dst(compressed_pos, intv_radius);
	size_t unpredictable_count = unpred_data.size();
	write_variable_to_dst(compressed_pos, unpredictable_count);
	write_array_to_dst(compressed_pos, (T_data *)&unpred_data[0], unpredictable_count);	
	size_t eb_quant_num = eb_quant_index_pos - eb_quant_index;
	write_variable_to_dst(compressed_pos, eb_quant_num);
	Huffman_encode_tree_and_data(2*1024, eb_quant_index, eb_quant_num, compressed_pos);
	free(eb_quant_index);
	size_t data_quant_num = data_quant_index_pos - data_quant_index;
	write_variable_to_dst(compressed_pos, data_quant_num);
	Huffman_encode_tree_and_data(2*capacity, data_quant_index, data_quant_num, compressed_pos);

	free(data_quant_index);
	compressed_size = compressed_pos - compressed;
	return compressed;	
}

template
unsigned char *
sz_compress_cp_preserve_sos_3d_online_fp_parallel(const float * U, const float * V, const float * W, size_t r1, size_t r2, size_t r3, size_t& compressed_size, bool transpose, double max_pwr_eb);

template<typename T_data>
unsigned char *
sz_compress_cp_preserve_sos_3d_online_fp_parallel_lossless_border(const T_data * U, const T_data * V, const T_data * W, size_t r1, size_t r2, size_t r3, size_t& compressed_size, bool transpose, double max_pwr_eb){
	// std::cout << "sz_compress_cp_preserve_sos_3d_online_fp" << std::endl;
	using T = int64_t;
	size_t num_elements = r1 * r2 * r3;
	T * U_fp = (T *) malloc(num_elements*sizeof(T));
	T * V_fp = (T *) malloc(num_elements*sizeof(T));
	T * W_fp = (T *) malloc(num_elements*sizeof(T));
	T range = 0;
	T vector_field_scaling_factor = convert_to_fixed_point(U, V, W, num_elements, U_fp, V_fp, W_fp, range);
	// printf("fixed point range = %lld\n", range);
	int * eb_quant_index = (int *) malloc(num_elements*sizeof(int));
	int * data_quant_index = (int *) malloc(3*num_elements*sizeof(int));
	int * eb_quant_index_pos = eb_quant_index;
	int * data_quant_index_pos = data_quant_index;
	// next, row by row
	const int base = 2;
	const double log_of_base = log2(base);
	const int capacity = 65536;
	const int intv_radius = (capacity >> 1);
	T max_eb = range * max_pwr_eb;
	unpred_vec<T_data> unpred_data;
	ptrdiff_t dim0_offset = r2 * r3;
	ptrdiff_t dim1_offset = r3;
	ptrdiff_t cell_dim0_offset = (r2-1) * (r3-1);
	ptrdiff_t cell_dim1_offset = r3-1;
	int simplex_offset[24];
	int index_offset[24][3][3];
	int offset[24][3];
	compute_offset(dim0_offset, dim1_offset, cell_dim0_offset, cell_dim1_offset, simplex_offset, index_offset, offset);
	T * cur_U_pos = U_fp;
	T * cur_V_pos = V_fp;
	T * cur_W_pos = W_fp;
	T threshold = 1;
	// check cp for all cells
	// std::cout << "start cp checking\n";
	vector<bool> cp_exist = compute_cp(U_fp, V_fp, W_fp, r1, r2, r3);
	// std::cout << "start compression\n";
	for(int i=0; i<r1; i++){
		for(int j=0; j<r2; j++){
			for(int k=0; k<r3; k++){
				T required_eb = max_eb;
				if(k == 0 || k == r3-1 || j == 0 || j == r2-1 || i == 0 || i == r1-1){
					required_eb = 0;
				}
				else{
					// derive eb given 24 adjacent simplex
					for(int n=0; n<24; n++){
						bool in_mesh = true;
						for(int p=0; p<3; p++){
							// reversed order!
							if(!(in_range(i + index_offset[n][p][2], (int)r1) && in_range(j + index_offset[n][p][1], (int)r2) && in_range(k + index_offset[n][p][0], (int)r3))){
								in_mesh = false;
								break;
							}
						}
						if(in_mesh){
							int index = simplex_offset[n] + 6*(i*(r2-1)*(r3-1) + j*(r3-1) + k);
							if(cp_exist[index]){
								required_eb = 0;
								break;
							}
							required_eb = MINF(required_eb, derive_cp_abs_eb_sos_online(
								cur_U_pos[offset[n][0]], cur_U_pos[offset[n][1]], cur_U_pos[offset[n][2]], *cur_U_pos,
								cur_V_pos[offset[n][0]], cur_V_pos[offset[n][1]], cur_V_pos[offset[n][2]], *cur_V_pos,
								cur_W_pos[offset[n][0]], cur_W_pos[offset[n][1]], cur_W_pos[offset[n][2]], *cur_W_pos));
						}
					}
				}
				T abs_eb = required_eb;
				*eb_quant_index_pos = eb_exponential_quantize(abs_eb, base, log_of_base, threshold);
				if(abs_eb > 0){
					bool unpred_flag = false;
					T decompressed[3];
					// compress vector fields
					T * data_pos[3] = {cur_U_pos, cur_V_pos, cur_W_pos};
					for(int p=0; p<3; p++){
						T * cur_data_pos = data_pos[p];
						T cur_data = *cur_data_pos;
						// get adjacent data and perform Lorenzo
						/*
							d6	X
							d4	d5
							d2	d3
							d0	d1
						*/
						T d0 = (i && j && k) ? cur_data_pos[- dim0_offset - dim1_offset - 1] : 0;
						T d1 = (i && j) ? cur_data_pos[- dim0_offset - dim1_offset] : 0;
						T d2 = (i && k) ? cur_data_pos[- dim0_offset - 1] : 0;
						T d3 = (i) ? cur_data_pos[- dim0_offset] : 0;
						T d4 = (j && k) ? cur_data_pos[- dim1_offset - 1] : 0;
						T d5 = (j) ? cur_data_pos[- dim1_offset] : 0;
						T d6 = (k) ? cur_data_pos[- 1] : 0;
						T pred = d0 + d3 + d5 + d6 - d1 - d2 - d4;
						T diff = cur_data - pred;
						T quant_diff = std::abs(diff) / abs_eb + 1;
						if(quant_diff < capacity){
							quant_diff = (diff > 0) ? quant_diff : -quant_diff;
							int quant_index = (int)(quant_diff/2) + intv_radius;
							data_quant_index_pos[p] = quant_index;
							decompressed[p] = pred + 2 * (quant_index - intv_radius) * abs_eb; 
							// check original data
							if(std::abs(decompressed[p] - cur_data) >= required_eb){
								unpred_flag = true;
								break;
							}
						}
						else{
							unpred_flag = true;
							break;
						}
					}
					if(unpred_flag){
						*(eb_quant_index_pos ++) = 0;
						ptrdiff_t offset = cur_U_pos - U_fp;
						unpred_data.push_back(U[offset]);
						unpred_data.push_back(V[offset]);
						unpred_data.push_back(W[offset]);
					}
					else{
						eb_quant_index_pos ++;
						data_quant_index_pos += 3;
						*cur_U_pos = decompressed[0];
						*cur_V_pos = decompressed[1];
						*cur_W_pos = decompressed[2];
					}
				}
				else{
					// record as unpredictable data
					*(eb_quant_index_pos ++) = 0;
					ptrdiff_t offset = cur_U_pos - U_fp;
					unpred_data.push_back(U[offset]);
					unpred_data.push_back(V[offset]);
					unpred_data.push_back(W[offset]);
				}
				cur_U_pos ++, cur_V_pos ++, cur_W_pos ++;
			}
		}
	}
	free(U_fp);
	free(V_fp);
	free(W_fp);
	// printf("offset eb_q, data_q, unpred: %ld %ld %ld\n", eb_quant_index_pos - eb_quant_index, data_quant_index_pos - data_quant_index, unpred_data.size());
	unsigned char * compressed = (unsigned char *) malloc(3*num_elements*sizeof(T));
	unsigned char * compressed_pos = compressed;
	write_variable_to_dst(compressed_pos, vector_field_scaling_factor);
	write_variable_to_dst(compressed_pos, base);
	write_variable_to_dst(compressed_pos, threshold);
	write_variable_to_dst(compressed_pos, intv_radius);
	size_t unpredictable_count = unpred_data.size();
	write_variable_to_dst(compressed_pos, unpredictable_count);
	write_array_to_dst(compressed_pos, (T_data *)&unpred_data[0], unpredictable_count);	
	size_t eb_quant_num = eb_quant_index_pos - eb_quant_index;
	write_variable_to_dst(compressed_pos, eb_quant_num);
	Huffman_encode_tree_and_data(2*1024, eb_quant_index, eb_quant_num, compressed_pos);
	free(eb_quant_index);
	size_t data_quant_num = data_quant_index_pos - data_quant_index;
	write_variable_to_dst(compressed_pos, data_quant_num);
	Huffman_encode_tree_and_data(2*capacity, data_quant_index, data_quant_num, compressed_pos);
	// printf("pos = %ld\n", compressed_pos - compressed);
	free(data_quant_index);
	compressed_size = compressed_pos - compressed;
	return compressed;	
}

template
unsigned char *
sz_compress_cp_preserve_sos_3d_online_fp_parallel_lossless_border(const float * U, const float * V, const float * W, size_t r1, size_t r2, size_t r3, size_t& compressed_size, bool transpose, double max_pwr_eb);
