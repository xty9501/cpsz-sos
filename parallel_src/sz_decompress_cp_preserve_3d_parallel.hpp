#include "sz_decompress_cp_preserve_3d.hpp"
#include "sz_decompress_block_processing.hpp"
#include <limits>
#include <unordered_set>

template<typename T, typename T_fp>
static void 
convert_to_floating_point(const T_fp * U_fp, const T_fp * V_fp, const T_fp * W_fp, size_t num_elements, T * U, T * V, T * W, int64_t vector_field_scaling_factor){
	for(int i=0; i<num_elements; i++){
		U[i] = U_fp[i] * (T)1.0 / vector_field_scaling_factor;
		V[i] = V_fp[i] * (T)1.0 / vector_field_scaling_factor;
		W[i] = W_fp[i] * (T)1.0 / vector_field_scaling_factor;
	}
}

template<typename T_data>
void
sz_decompress_cp_preserve_3d_online_fp_parallel(const unsigned char * compressed, size_t r1, size_t r2, size_t r3, T_data *& U, T_data *& V, T_data *& W){
	if(U) free(U);
	if(V) free(V);
	if(W) free(W);
	using T = int64_t;
	size_t num_elements = r1 * r2 * r3;
	const unsigned char * compressed_pos = compressed;
	T vector_field_scaling_factor = 0;
	read_variable_from_src(compressed_pos, vector_field_scaling_factor);
	int base = 0;
	read_variable_from_src(compressed_pos, base);
	// printf("base = %d\n", base);
	T threshold = 0;
	read_variable_from_src(compressed_pos, threshold);
	int intv_radius = 0;
	read_variable_from_src(compressed_pos, intv_radius);
	const int capacity = (intv_radius << 1);
	size_t unpred_data_count = 0;
	read_variable_from_src(compressed_pos, unpred_data_count);
	const T_data * unpred_data = (T_data *) compressed_pos;
	const T_data * unpred_data_pos = unpred_data;
	compressed_pos += unpred_data_count*sizeof(T_data);
	size_t eb_quant_num = 0;
	read_variable_from_src(compressed_pos, eb_quant_num);
	int * eb_quant_index = Huffman_decode_tree_and_data(2*1024, eb_quant_num, compressed_pos);
	size_t data_quant_num = 0;
	read_variable_from_src(compressed_pos, data_quant_num);
	int * data_quant_index = Huffman_decode_tree_and_data(2*capacity, data_quant_num, compressed_pos);
	// printf("pos = %ld\n", compressed_pos - compressed);
	T * U_fp = (T *) malloc(num_elements*sizeof(T));
	T * V_fp = (T *) malloc(num_elements*sizeof(T));
	T * W_fp = (T *) malloc(num_elements*sizeof(T));
	T * U_pos = U_fp;
	T * V_pos = V_fp;
	T * W_pos = W_fp;
	size_t dim0_offset = r2 * r3;
	size_t dim1_offset = r3;
	int * eb_quant_index_pos = eb_quant_index;
	int * data_quant_index_pos = data_quant_index;
	std::vector<int> unpred_data_indices;
	// std::cout << "decompress cube\n"; 
	for(int i=0; i<r1-1; i++){
		for(int j=0; j<r2-1; j++){
			for(int k=0; k<r3-1; k++){
				// printf("%ld %ld %ld\n", i, j, k);
				T * data_pos[3] = {U_pos, V_pos, W_pos};
				// get eb
				if(*eb_quant_index_pos == 0){
					size_t offset = U_pos - U_fp;
					unpred_data_indices.push_back(offset);
					*U_pos = *(unpred_data_pos ++) * vector_field_scaling_factor;
					*V_pos = *(unpred_data_pos ++) * vector_field_scaling_factor;
					*W_pos = *(unpred_data_pos ++) * vector_field_scaling_factor;
					eb_quant_index_pos ++;
				}
				else{
					T eb = pow(base, *eb_quant_index_pos ++) * threshold;
					for(int p=0; p<3; p++){
						T * cur_data_pos = data_pos[p];					
						T d0 = (i && j && k) ? cur_data_pos[- dim0_offset - dim1_offset - 1] : 0;
						T d1 = (i && j) ? cur_data_pos[- dim0_offset - dim1_offset] : 0;
						T d2 = (i && k) ? cur_data_pos[- dim0_offset - 1] : 0;
						T d3 = (i) ? cur_data_pos[- dim0_offset] : 0;
						T d4 = (j && k) ? cur_data_pos[- dim1_offset - 1] : 0;
						T d5 = (j) ? cur_data_pos[- dim1_offset] : 0;
						T d6 = (k) ? cur_data_pos[- 1] : 0;
						T pred = d0 + d3 + d5 + d6 - d1 - d2 - d4;
						*cur_data_pos = pred + 2 * (data_quant_index_pos[p] - intv_radius) * eb;
					}
					data_quant_index_pos += 3;
				}
				U_pos ++, V_pos ++, W_pos ++;
			}
			// skip the last element
			// TODO: adjust for parallelization
			U_pos ++, V_pos ++, W_pos ++;
		}
		// skip the last line
		// TODO: adjust for parallelization
		U_pos += dim1_offset, V_pos +=dim1_offset, W_pos +=dim1_offset;
	}
	// decompress top surface (r2-1)*(r3-1)
	{
		// std::cout << "decompress top\n";
		int i = r1 - 1;
		for(int j=0; j<r2-1; j++){
			U_pos = U_fp + i*dim0_offset + j*dim1_offset, V_pos = V_fp + i*dim0_offset + j*dim1_offset, W_pos = W_fp + i*dim0_offset + j*dim1_offset;
			for(int k=0; k<r3-1; k++){
				// printf("%ld %ld %ld\n", i, j, k);
				T * data_pos[3] = {U_pos, V_pos, W_pos};
				// get eb
				if(*eb_quant_index_pos == 0){
					size_t offset = U_pos - U_fp;
					unpred_data_indices.push_back(offset);
					*U_pos = *(unpred_data_pos ++) * vector_field_scaling_factor;
					*V_pos = *(unpred_data_pos ++) * vector_field_scaling_factor;
					*W_pos = *(unpred_data_pos ++) * vector_field_scaling_factor;
					eb_quant_index_pos ++;
				}
				else{
					T eb = pow(base, *eb_quant_index_pos ++) * threshold;
					for(int p=0; p<3; p++){
						T * cur_data_pos = data_pos[p];					
						T d0 = cur_data_pos[- dim0_offset - dim1_offset - 1];
						T d1 = cur_data_pos[- dim0_offset - dim1_offset];
						T d2 = cur_data_pos[- dim0_offset - 1];
						T d3 = cur_data_pos[- dim0_offset];
						T d4 = cur_data_pos[- dim1_offset - 1];
						T d5 = cur_data_pos[- dim1_offset];
						T d6 = cur_data_pos[- 1];
						T pred = d0 + d3 + d5 + d6 - d1 - d2 - d4;
						*cur_data_pos = pred + 2 * (data_quant_index_pos[p] - intv_radius) * eb;
					}
					data_quant_index_pos += 3;
				}
				U_pos ++, V_pos ++, W_pos ++;
			}
		}
	}
	// decompress right surface r1*(r2-1)
	{
		// std::cout << "decompress right\n";
		int k = r3-1;
		for(int i=0; i<r1; i++){
			U_pos = U_fp + i*dim0_offset + k, V_pos = V_fp + i*dim0_offset + k, W_pos = W_fp + i*dim0_offset + k;
			for(int j=0; j<r2-1; j++){
				// printf("%ld %ld %ld\n", i, j, k);
				T * data_pos[3] = {U_pos, V_pos, W_pos};
				// get eb
				if(*eb_quant_index_pos == 0){
					size_t offset = U_pos - U_fp;
					unpred_data_indices.push_back(offset);
					*U_pos = *(unpred_data_pos ++) * vector_field_scaling_factor;
					*V_pos = *(unpred_data_pos ++) * vector_field_scaling_factor;
					*W_pos = *(unpred_data_pos ++) * vector_field_scaling_factor;
					eb_quant_index_pos ++;
				}
				else{
					T eb = pow(base, *eb_quant_index_pos ++) * threshold;
					for(int p=0; p<3; p++){
						T * cur_data_pos = data_pos[p];					
						T d0 = cur_data_pos[- dim0_offset - dim1_offset - 1];
						T d1 = cur_data_pos[- dim0_offset - dim1_offset];
						T d2 = cur_data_pos[- dim0_offset - 1];
						T d3 = cur_data_pos[- dim0_offset];
						T d4 = cur_data_pos[- dim1_offset - 1];
						T d5 = cur_data_pos[- dim1_offset];
						T d6 = cur_data_pos[- 1];
						T pred = d0 + d3 + d5 + d6 - d1 - d2 - d4;
						*cur_data_pos = pred + 2 * (data_quant_index_pos[p] - intv_radius) * eb;
					}
					data_quant_index_pos += 3;
				}
				U_pos += dim1_offset, V_pos += dim1_offset, W_pos += dim1_offset;
			}
		}
	}
	// decompress back surface r1*r3
	{
		// std::cout << "decompress back\n";
		int j = r2-1;
		for(int i=0; i<r1; i++){
			U_pos = U_fp + i*dim0_offset + j*dim1_offset, V_pos = V_fp + i*dim0_offset + j*dim1_offset, W_pos = W_fp + i*dim0_offset + j*dim1_offset;
			for(int k=0; k<r3; k++){
				// printf("%ld %ld %ld\n", i, j, k);
				T * data_pos[3] = {U_pos, V_pos, W_pos};
				// get eb
				if(*eb_quant_index_pos == 0){
					size_t offset = U_pos - U_fp;
					unpred_data_indices.push_back(offset);
					*U_pos = *(unpred_data_pos ++) * vector_field_scaling_factor;
					*V_pos = *(unpred_data_pos ++) * vector_field_scaling_factor;
					*W_pos = *(unpred_data_pos ++) * vector_field_scaling_factor;
					eb_quant_index_pos ++;
				}
				else{
					T eb = pow(base, *eb_quant_index_pos ++) * threshold;
					for(int p=0; p<3; p++){
						T * cur_data_pos = data_pos[p];					
						T d0 = cur_data_pos[- dim0_offset - dim1_offset - 1];
						T d1 = cur_data_pos[- dim0_offset - dim1_offset];
						T d2 = cur_data_pos[- dim0_offset - 1];
						T d3 = cur_data_pos[- dim0_offset];
						T d4 = cur_data_pos[- dim1_offset - 1];
						T d5 = cur_data_pos[- dim1_offset];
						T d6 = cur_data_pos[- 1];
						T pred = d0 + d3 + d5 + d6 - d1 - d2 - d4;
						*cur_data_pos = pred + 2 * (data_quant_index_pos[p] - intv_radius) * eb;
					}
					data_quant_index_pos += 3;
				}
				U_pos ++, V_pos ++, W_pos ++;
			}
		}
	}
	// printf("recover data done\n");
	free(eb_quant_index);
	free(data_quant_index);
	U = (T_data *) malloc(num_elements*sizeof(T_data));
	V = (T_data *) malloc(num_elements*sizeof(T_data));
	W = (T_data *) malloc(num_elements*sizeof(T_data));
	convert_to_floating_point(U_fp, V_fp, W_fp, num_elements, U, V, W, vector_field_scaling_factor);
	unpred_data_pos = unpred_data;
	for(const auto& index:unpred_data_indices){
		U[index] = *(unpred_data_pos++);
		V[index] = *(unpred_data_pos++);
		W[index] = *(unpred_data_pos++);
	}
	free(U_fp);
	free(V_fp);
	free(W_fp);
}

template
void
sz_decompress_cp_preserve_3d_online_fp_parallel<float>(const unsigned char * compressed, size_t r1, size_t r2, size_t r3, float *& U, float *& V, float *& W);
