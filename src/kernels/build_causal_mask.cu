#include "src/kernels/build_casual_mask.h"

template<typename T>
__global__ void BuildCausalMasksConsideringContextPastKV(T* mask,
                                                const int* q_lens,  //input lens, shape=[batch size]
                                                const int* k_lens,  //context lens, shape=[batch size]
                                                int max_q_len,  // max(q_lens)
                                                int max_k_len){ // max(k_lens)
                                                
    int tid = threadIdx.x;
    int qlen = q_lens[blockIdx.x];
    int klen = k_lens[blockIdx.x];

    mask += blockIdx.x * max_q_len*max_k_len;

    while(tid < max_q_len*max_k_len){
        int q = tid / max_k_len;
        int k = tid % max_k_len;

        bool is_one = q < qlen && k < klen && k <= q + (klen - qlen) && k >= klen - qlen;
        mask[tid] = static_cast<T>(is_one);

        // 保证遍历完一个bs中所有的空间
        tid += blockDim.x;
    }
}

template<typename T>
void launchBuildCausalMasks(TensorWrapper<T>* mask, 
                            TensorWrapper<int>* q_lens, 
                            TensorWrapper<int>* k_lens)
{
    int batch_size = mask->shape[0];
    int max_q_len = mask->shape[1];
    int max_k_len = mask->shape[2];
    // XuLin-1017: 此处的max_q_len和max_k_len是经过统计后得出的外部输入
    BuildCausalMasksConsideringContextPastKV<T><<<batch_size, 256>>>(mask->data, q_lens->data, k_lens->data, max_q_len, max_k_len);
}

template void launchBuildCausalMasks(TensorWrapper<float>* mask, 
                            TensorWrapper<int>* q_lens, 
                            TensorWrapper<int>* k_lens);

template void launchBuildCausalMasks(TensorWrapper<half>* mask, 
                            TensorWrapper<int>* q_lens, 
                            TensorWrapper<int>* k_lens);