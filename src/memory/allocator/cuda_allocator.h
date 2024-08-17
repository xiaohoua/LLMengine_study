#pragma once
#include <unordered_map>
#include <map>
#include <vector>
#include <iostream>
#include "src/memory/allocator/base_allocator.h"
#include "src/utils/macro.h"

// I use Bytes to printf buffer size msg, because sometime I allocate <1KB buffer, which causes that display 0KB
// 这里分了两种块 
struct CudaBigBlock {
    void *data;
    size_t size;
    bool is_allocated;

    CudaBigBlock() = default;
    CudaBigBlock(void* data_, int size_, bool is_allocated_):
        data(data_),
        size(size_),
        is_allocated(is_allocated_){}
};

struct CudaSmallBlock {
    void *data;
    size_t size;
    bool is_allocated;

    CudaSmallBlock() = default;
    CudaSmallBlock(void* data_, int size_, bool is_allocated_):
        data(data_),
        size(size_),
        is_allocated(is_allocated_){}
};


class CudaAllocator: public BaseAllocator{
private:
    std::map<int, std::vector<CudaSmallBlock>> cudaSmallBlocksMap;
    std::map<int, std::vector<CudaBigBlock>> cudaBigBlocksMap;
    int dev_id;
    std::map<int, size_t> FreeSize;  
    size_t total_allocated_size = 0;  

public:
    CudaAllocator() {
        cudaGetDevice(&dev_id);
    }
    ~CudaAllocator() {
        for (auto &it: cudaSmallBlocksMap) {
            auto &cudaBlocks = it.second; //vector
            for (int i = 0; i < cudaBlocks.size(); i++) {
                cudaFree(cudaBlocks[i].data);
            }
            auto &bigBlocks = cudaBigBlocksMap[it.first];
            for (int i = 0; i < bigBlocks.size(); i++) {
                cudaFree(bigBlocks[i].data);
            }            
        }
    }
    void* UnifyMalloc(void* ptr, size_t size, bool is_host){
        size = ((size + 31) / 32) * 32;
        //1.host malloc
        if(is_host){
            ptr = malloc(size);
            memset(ptr, 0, size);//逐字节初始化0,初始化size个字节
            return ptr; //不需要频繁申请cpu内存，只在用户输入句子那里需要
        }
        //2.big buf 先在bigblocks里找空闲的（free除了还没还给os的）
        if(size > 1024*1024){
            //只要大于1024K就视为大buf
            auto BigBlocks = cudaBigBlocksMap[dev_id];
            int blockID = -1;
            for(int i = 0; i < BigBlocks.size(); i++){
                if(BigBlocks[i].size() > size && BigBlocks[i].is_allocated
                    && BigBlocks[i].size - size < 1024*1024){
                        //第三个判断条件是从pool里拿到的内存空间不能必须要的大太多
                        //不然会有内存碎片
                        if(blockID == -1 || BigBlocks[blockID].size > BigBlocks[i].size)
                        {//找到最小的存放空间
                            blockID = i;
                        }
                    }
            }
            if(blockID != -1){
                BigBlocks[blockID].is_allocated = true;
                return BigBlocks[blockID].data;
            }
            //没找到
            void* new_buffer;
            cudaMalloc(&new_buffer, size);
            total_allocated_size += size;
            // std::cout << "allocate a new big block from OS using cudaMalloc, size = "
            //                                     << size << "B, total allocated size " << total_allocated_size << "B"
            //                                     << std::endl;
            BigBlocks.push_back(CudaBigBlock(new_buffer, size, true));
            return new_buffer;
        }
        
        //3.small buf 先去smallblocks里找空闲的
        //用 free
        auto &SmallBlocks = cudaSmallBlocksMap[dev_id];
        for (int i = 0; i < SmallBlocks.size(); i++) {
            if (SmallBlocks[i].size >= size && !SmallBlocks[i].is_allocated) {
                SmallBlocks[i].is_allocated = true;
                FreeSize[i] += SmallBlocks[i].size;//小buf size
                return SmallBlocks[i].data;
            }
        }
        // 4.没找到空闲的再cudaMalloc
        void* new_buffer = (void*)ptr;
        CHECK(cudaMalloc(&new_buffer, size));
        CHECK(cudaMemset(new_buffer, 0, size));
        // std::cout << "allocate a new small block from OS using cudaMalloc, size = "
        //                                     << size  << "B, total allocated size " << total_allocated_size << "B"
        //                                     << std::endl;

        SmallBlocks.push_back(CudaSmallBlock(new_buffer, size, true));
        return new_buffer;
    }

    void UnifyFree(void* ptr, bool is_host) {
        if (ptr == nullptr) {
            return;
        }
        // 1.host free
        if (is_host) {
            free(ptr);
            return;
        }
        // 2.清理碎片：当累计的小buf超出了1G时，清理未分配出去的smallblocks, 已分配的还是保留在smallmap
        for (auto &it: cudaSmallBlocksMap) {
            if (FreeSize[it.first] > 1024 * 1024 * 1024) {
                auto &cudaBlocks = it.second;
                std::vector<CudaSmallBlock> temp;
                for (int i = 0; i < cudaBlocks.size(); i++) {
                    if (!cudaBlocks[i].is_allocated) {
                        cudaSetDevice(it.first);
                        // std::cout << "free a small block to OS using cudaFree, block id = "
                        //                                     << i
                        //                                     << ",size = "
                        //                                     << cudaBlocks[i].size << "B"
                        //                                     << std::endl;
                        cudaFree(cudaBlocks[i].data);
                    } else {
                        temp.push_back(cudaBlocks[i]);
                    }
                }
                cudaBlocks.clear();
                it.second = temp;
                FreeSize[it.first] = 0;
            }
        }
        // 3.找到待free的buffer的位置，设is_allocated = false，大小block都不归还到OS，除非没有在大小block里面找到待free的ptr
        // 大块清理分配比较耗时，为了降低损耗，用标记为标记为已经清除即可
        for (auto &it: cudaSmallBlocksMap) {
            auto &cudaBlocks = it.second;
            for (int i = 0; i < cudaBlocks.size(); i++) {
                if (cudaBlocks[i].data == ptr) {
                    FreeSize[it.first] += cudaBlocks[i].size;
                    cudaBlocks[i].is_allocated = false;
                    // std::cout << "free a small block but not to OS, block id = "
                    //                                     << i
                    //                                     << ",size = "
                    //                                     << cudaBlocks[i].size << "B"
                    //                                     << std::endl;
                    return;
                }
            }
            //若是大block，那不归还到OS
            auto &bigBlocks = cudaBigBlocksMap[it.first];
            for (int i = 0; i < bigBlocks.size(); i++) {
                if (bigBlocks[i].data == ptr) {
                    // std::cout << "free a big block but not to OS, block id = "
                    //                                     << i
                    //                                     << ",size = "
                    //                                     << cudaBlocks[i].size << "B"
                    //                                     << std::endl;
                    bigBlocks[i].is_allocated = false;
                    return;
                }
            }
        }
        // std::cout << "NOT found the ptr in blocks, so free the ptr to OS using cudaFree"
                                            // << std::endl;
        cudaFree(ptr);    
    }
};
