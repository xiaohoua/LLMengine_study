add_executable(embedding
    test_input_embedding.cu
)
target_link_libraries(
    embedding PUBLIC
    -lcudart
    -lcudadevrt
    embeddingFunctor)

add_executable(paddingoffset
    test_cal_paddingoffset.cu
)
target_link_libraries(
    paddingoffset PUBLIC
    -lcudart
    -lcudadevrt
    paddingoffsetFunctor)

add_executable(rms_norm
    test_rmsnorm.cu
)
target_link_libraries(
    rms_norm PUBLIC
    -lcudart
    -lcudadevrt
    rmsnormFunctor)

add_executable(causal_mask
    test_causal_mask.cu
)
target_link_libraries(
    causal_mask PUBLIC
    -lcudart
    -lcudadevrt
    BuildCausalMaskFunctor)

    add_executable(testlinear
    test_linear.cu
)
target_link_libraries(  # Libs for test_build_casual_mask
    testlinear PUBLIC
    -lcudart
    -lcudadevrt
    linear) 

    add_executable(biasRope
    test_bias_and_RoPE.cu
)
target_link_libraries(  # Libs for test_qkv_bias_and_rope
  biasRope PUBLIC
    -lcudart
    -lcudadevrt
    qkv_bias_and_rope) 

    add_executable(test_concat_kv
    test_concat_kv.cu
)
target_link_libraries(  
    test_concat_kv PUBLIC
    -lcudart
    -lcudadevrt
    concat_kv) 