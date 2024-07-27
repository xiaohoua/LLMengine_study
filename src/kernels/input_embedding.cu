#include<stdio.h>
#include"src/kernels/input_embedding.h"

template<typename T>
__global__ void embeddingFunctor(const int* input_ids,
                T* output,
                const T* embed_table,
                const int max_context_token_num,
                const int hidden_size)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    while (index < max_context_token_num*hidden_size)
    {
        int id = input_ids[index / hidden_size];
        output[index] = embed_table[id * hidden_size + index % hidden_size];
        index += blockDim.x * gridDim.x;
    }
}

template<typename T>
void launchInputEmbedding(TensorWrapper<int>* input_ids,
                            TensorWrapper<T>* output,
                            EmbeddingWeight<T>* embed_table)
{
    const int blockSize = 256;
    const int max_context_token_num = output->shape[0];
    const int hidden_size = output->shape[1];
    const int gridSize = 2048;
    LLM_CHECK_WITH_INFO(max_context_token_num == input_ids->shape[0],
                        "input ids 1st shape should equal to 1st shape of output");
    embeddingFunctor<<<gridSize, blockSize>>>(input_ids->data,
                                                output->data,
                                                embed_table->data,
                                                max_context_token_num,
                                                hidden_size);
    #ifdef PRINT_DATA
        print_data<<<1, 1>>>(output->data);
    #else
    #endif
}
// zhaziqwe: 显式实例化模版函数，由于cuda的语法规则，不能存在.cpp文件里，因此只能在此实例化
template void launchInputEmbedding(TensorWrapper<int>* input_ids,    
                                   TensorWrapper<float>* output,       
                                   EmbeddingWeight<float>* embed_table);
template void launchInputEmbedding(TensorWrapper<int>* input_ids,    
                                   TensorWrapper<half>* output,       
                                   EmbeddingWeight<half>* embed_table);