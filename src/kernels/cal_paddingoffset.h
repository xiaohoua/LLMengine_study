#include"src/utils/tensor.h"
#include"src/utils/macro.h"
void launchCalPaddingoffset(TensorWrapper<int>* paddingoffset,
                            TensorWrapper<int>* cum_seqlens,
                            TensorWrapper<int>* input_length)