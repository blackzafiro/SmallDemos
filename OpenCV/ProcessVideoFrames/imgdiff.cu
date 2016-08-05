#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>


// http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __func__, __LINE__, false); }
inline void gpuAssert(cudaError_t code, const char *file, const char *func, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"\e[1;31mGPUassert: %s %s %s %d\n\e[1;0m", cudaGetErrorString(code), file, func, line);
      if (abort) exit(code);
   }
}

// http://stackoverflow.com/questions/24613637/custom-kernel-gpumat-with-float

 __global__ void gpuNumDifferentKernel(const cv::cuda::PtrStepSz<uchar4> img1,
                                cv::cuda::PtrStepSz<uchar4> img2,
                                int* d_diff)
{
	// which thread is this and in which block is it?
	// These variables won't be used, but they are here for illustrative purposes
	// http://www.martinpeniak.com/index.php?option=com_content&view=article&catid=17:updates&id=288:cuda-thread-indexing-explained
	int blockId = blockIdx.x + blockIdx.y * gridDim.x; 
	int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	/*if(blockIdx.x < 20 && blockIdx.y < 10)
	printf("Coord (%d, %d) thread(%d, %d, %d) bldim(%d, %d, %d) block(%d, %d, %d) grdim(%d, %d, %d)\n", y, x,
	       threadIdx.x, threadIdx.y, threadIdx.z, blockDim.x, blockDim.y, blockDim.z,
	       blockIdx.x, blockIdx.y, blockIdx.z, gridDim.x, gridDim.y, gridDim.z);*/

	__shared__ int sh_diff;

	// One thread per block mush initialize the shared memory for that block
	if(threadIdx.x == 0 && threadIdx.y == 0) {
		sh_diff = 0;
	}

	__syncthreads();

	if (x < img1.cols && y < img1.rows && y >= 0 && x >= 0)
	{
		if(img1(y, x).x != img2(y, x).x ||
		   img1(y, x).y != img2(y, x).y ||
		   img1(y, x).z != img2(y, x).z)
		{
			atomicAdd(&sh_diff, 1);
		}
		img2(y, x).z = 0;
	}

	__syncthreads();

	// Same thread adds the contribution of the block to global count
	if(threadIdx.x == 0 && threadIdx.y == 0)
	{
		atomicAdd(d_diff, sh_diff);
	}

}

/**
 * Demo function to illustrate how to create a new kernel for GpuMat,
 * it counts the number of pixels with different value beetween
 * img1 and img2.
 */
int gpuNumDifferent(cv::InputArray _img1,
                     cv::OutputArray _img2,
                     cv::cuda::Stream _stream)
{
	const cv::cuda::GpuMat img1 = _img1.getGpuMat();
	const cv::cuda::GpuMat img2 = _img2.getGpuMat();

	dim3 cthreads_blockDim(32, 32);
	dim3 cblocks_gridDim(
		static_cast<int>(std::ceil(img1.size().width /
		    static_cast<double>(cthreads_blockDim.x))),
		static_cast<int>(std::ceil(img1.size().height / 
		    static_cast<double>(cthreads_blockDim.y))));

	int h_diff = 0;
	int *d_diff;

	gpuErrchk( cudaMalloc((void**)&d_diff, sizeof(int)) );
	gpuErrchk( cudaMemcpy((void*)d_diff, (void*)&h_diff, sizeof(int), cudaMemcpyHostToDevice) );

	cudaStream_t stream = cv::cuda::StreamAccessor::getStream(_stream);
	gpuNumDifferentKernel<<<cblocks_gridDim, cthreads_blockDim, 0, stream>>>(img1, img2, d_diff);

	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	gpuErrchk( cudaMemcpy((void*)&h_diff, (void*)d_diff, sizeof(int), cudaMemcpyDeviceToHost) );

	return h_diff;
}