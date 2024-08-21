#include <hip/hip_runtime.h>
#include <type_traits>
#include <vector>
#include <iostream>
#include <string>
#include <algorithm>
#include <ctime>    // 包含 time 函数
#include <cstdlib>  // 包含 rand 和 srand 函数

template <typename scalar_t>
void debug_scalar(std::string str, scalar_t value)
{
    std::cout << str <<": " << value << std::endl; 
}

std::vector<float> CpuTopKKernel(const std::vector<float>& input, int K)
{
    std::vector<float>data(input);
    std::vector<float> output(K);
    int len = data.size();
    std::cout << len << std::endl;
    
    for (int i=0; i<len-1; i++)
    {
        for (int j=0; j<len-1-i; j++)
        {
            if (data[j] < data[j+1])
            {
                std::swap(data[j], data[j+1]);
            }
        }
    }

    std::copy(data.begin(), data.begin()+K, output.begin());
    return output;
}


__device__ void ReduceTopK(float* smem1, float* smem2, int K)
{
    for (int i=0; i<K; i++)
    {
        float cur = smem2[i];
        if (cur > smem1[K-1])
        {
            smem1[K-1] = cur;
            for (int j=K-2; j>=0; j--)
            {
                if (cur > smem1[j])
                {
                    float temp = smem1[j];
                    smem1[j] = smem1[j+1];
                    smem1[j+1]= temp;
                }
                else
                {
                    break;
                }
            }
        }

    }
}

__device__  void TopKInThread(const float* in_data, float* smem, int K, int num_ele_per_thread, int row_length) 
{
    int idx = threadIdx.x;
    // initialize shared memory
    for (int i=0; i<K; i++){
        smem[i] = -std::numeric_limits<float>::max();
    }
    __syncthreads();

    for (int i=0; i<num_ele_per_thread; i++)
    {
        if (idx*num_ele_per_thread + i < row_length){
            float cur = *(in_data + idx*num_ele_per_thread + i);
            // find the first element which is greater than or equal to cur.
            if (cur > smem[K-1])
            {
                smem[K-1] = cur;
                for (int j=K-2; j>=0; j--)
                {
                    if (cur >= smem[j])
                    {
                        float temp = smem[j];
                        smem[j] = smem[j+1];
                        smem[j+1]= temp;
                    }
                    else
                    {
                        break;
                    }
                }
            }

        }
    }
    __syncthreads();
}

__global__ void LaunchTopKKernel(const float* in, float* out, size_t K, size_t element_per_thread, size_t row_length)
{
    extern __shared__ float smem[];
    int idx = threadIdx.x;

    TopKInThread(in, smem+idx*K, K, element_per_thread, row_length);
    __syncthreads();

    for (int stride=1; stride < blockDim.x; stride=stride*2)
    {
        if (idx % (2*stride) == 0)
        {
            ReduceTopK(smem+idx*K, smem+(idx+stride)*K, K);
        }
        __syncthreads();
    }

    if (idx == 0)
    {
        for (int i=0; i<K; i++){
            out[i] = smem[i];
        }
    }
    // for (int i=0; i<K; i++){
    //     out[idx*K+i] = smem[idx*K+i];
    // }

    __syncthreads();
}


#define K 100
#define MAX(a, b) ((a) > (b) ? (a) : (b));
#define MIN(a, b) ((a) < (b) ? (a) : (b));

int main()
{
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    std::vector<float> input_data;
    size_t row_length = 10000;
    for (int i=0; i<row_length; i++)
    {
        input_data.emplace_back(std::rand() % 10000 + 1);
    }
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);

    // 获取每个block能够使用的最大的共享内存
    size_t max_smem = prop.sharedMemPerBlock;
    debug_scalar("max_smem", max_smem);

    // 根据K算出每个block能lanuch的,且能进行reduction最大线程数，需要同时满足3个条件
    // 1. 每个block的shared memory能够为这些线程分配足够的资源
    // 2. 线程的数量要是2的整数幂，这是为了能够正常的进行reduction操作
    // 3. 不能超过每个block的ALU的数量，这样是为了避免切换线程上下文
    size_t temp_thread_per_block = max_smem / (K * sizeof(float));
    size_t temp_thread_per_block_limit = std::pow(2, std::floor(std::log2(temp_thread_per_block)));
    size_t thread_per_block_limit = MIN(1024, temp_thread_per_block_limit); 
    debug_scalar<size_t>("thread_per_block_limit", thread_per_block_limit);
    

    // 如果按照最大的线程数来计算，根据元素总数算出每个线程应该处理多少数据， 这样算的好处是 每个线程处理尽量少的数据
    size_t temp_element_per_thread = (row_length + thread_per_block_limit -1) / thread_per_block_limit;
    size_t element_per_thread = MAX(K, temp_element_per_thread);

    // 按照每个元素应该处理的数据和元素的总数，再算出每个block的实际线程数，还应满足如下条件
    // 1. 为了正常进行reduction操作，线程数量应为2的整数次幂
    // 2. 不能超过上文中计算出的 thread_per_block_limit    size_t tmp_acutal_thread_per_block = (row_length + element_per_thread -1) / element_per_thread;
    size_t tmp_acutal_thread_per_block = (row_length + element_per_thread -1) / element_per_thread;
    size_t tmp2_acutal_thread_per_block = std::pow(2, std::ceil(std::log2(tmp_acutal_thread_per_block)));
    size_t actual_thread_per_block = MIN(tmp2_acutal_thread_per_block, thread_per_block_limit);
    debug_scalar("acutal_thread_per_block", actual_thread_per_block);


    // 为输出输出分配设备内存
    void* result_ptr, * in_ptr;
    hipMalloc(&result_ptr, K*sizeof(float));
    // hipMalloc(&result_ptr, acutal_thread_per_block*K*sizeof(float));
    hipMalloc(&in_ptr, row_length*sizeof(float));
    hipMemcpy(in_ptr, input_data.data(), row_length*sizeof(float), hipMemcpyHostToDevice);

    LaunchTopKKernel<<<1, actual_thread_per_block, K*actual_thread_per_block*sizeof(float)>>>((float*)in_ptr, (float*)result_ptr, K, element_per_thread, row_length);

    std::vector<float> output_data(K);
    hipMemcpy(output_data.data(), result_ptr, K*sizeof(float), hipMemcpyDeviceToHost);
    // hipMemcpy(output_data.data(), result_ptr, acutal_thread_per_block*K*sizeof(float), hipMemcpyDeviceToHost);

    std::vector<float> golden_res = CpuTopKKernel(input_data, K);

    std::cout << "input:" << std::endl;
    for (int i=0; i<row_length; i++)
    {
        std::cout << input_data[i] << ", ";
        if ((i+1) % 10 == 0)
        {
            std::cout << std::endl;
        }
    }

    // std::cout << "GPU output:" << std::endl;
    // for (int i=0; i<K*acutal_thread_per_block; i++)
    // {
    //     std::cout << output_data[i] << ", ";
    //     if ((i+1) % K == 0)
    //     {
    //         std::cout  << std::endl;
    //     }
    // }
    // std::cout << std::endl;

    std::cout << "GPU output:" << std::endl;
    for (int i=0; i<K; i++)
    {
        std::cout << output_data[i] << ", ";
    }
    std::cout << std::endl;


    std::cout << "cpu output:" << std::endl;
    for (int i=0; i<K; i++)
    {
        if (golden_res[i] != output_data[i])
        {
            std::cout << "Failed!" << std::endl;
            std::exit(-1);
        }
    }
        std::cout << "Successful!" << std::endl;

    std::cout << std::endl;

    return 0;
}



