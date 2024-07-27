#include"src/kernels/cal_paddingoffset.h"

__global__ void CalPaddingoffset(int*         padding_offset, 
                                int*         cum_seqlens,
                                const int*   input_lengths, //actual input lens
                                const int    batch_size,
                                const int    max_q_len) {
    
    int total_seqlen = 0;
    int cum_offset = 0;
    int index = 0;
    for(int i=0; i<batch_size; i++){
        int seqlen = input_lengths[i];
        cum_seqlens[i] = total_seqlen;
        total_seqlen += seqlen;
        for(int j=0; j<seqlen; j++){
            padding_offset[index] = cum_offset;
            index++;
        }
        cum_offset += max_q_len - seqlen;
    }
};

void launchCalPaddingoffset(TensorWrapper<int>* padding_offset,
                            TensorWrapper<int>* cum_seqlens,
                            TensorWrapper<int>* input_lengths)
{
    const int batch_size = padding_offset->shape[0];
    const int max_q_len = padding_offset->shape[1];
    LLM_CHECK_WITH_INFO(batch_size == input_lengths->shape[0],
                        "input length should equal to batch_size");
    LLM_CHECK_WITH_INFO(batch_size == cum_seqlens->shape[0],
                        "cum seqlen should equal to batch_size");
    CalPaddingoffset<<<1, 1>>>(
        padding_offset->data, cum_seqlens->data, input_lengths->data, batch_size, max_q_len
    );
};