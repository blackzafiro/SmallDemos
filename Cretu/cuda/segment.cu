#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include "GNG.h"
#include <math.h>       /* sqrt */


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

 __global__ void gpuNumDifferentKernel(int maxNumNodes,
								const cv::cuda::PtrStepSz<uchar4> d_img,
								GNG::HSV2DVector* d_nodes,
                                const cv::cuda::PtrStepSz<uchar> d_connections
                                )
{
	/*
	// which thread is this and in which block is it?
	// These variables won't be used, but they are here for illustrative purposes
	// http://www.martinpeniak.com/index.php?option=com_content&view=article&catid=17:updates&id=288:cuda-thread-indexing-explained
	int blockId = blockIdx.x + blockIdx.y * gridDim.x; 
	int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

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
	*/

}

/**
 * It segments the image using growing neural gas.
 * Receives an image in HSV color space which must fit in a single thread block.
 * Creates and returns a growing neural gas network on param **gng
 */
int gpuBuildGngForSegmentation(cv::InputArray _imgHsv,
					int maxNumNodes,
                    GNG::SegmentationGNG **gngPointer,
                    cv::cuda::Stream _stream)
{	
	cv::cuda::DeviceInfo dev_info = cv::cuda::DeviceInfo();
	cv::Vec<int, 3> blockSize = dev_info.maxThreadsDim();
	
	
	const cv::cuda::GpuMat d_img = _imgHsv.getGpuMat();
	
	GNG::HSV2DVector h_nodes[maxNumNodes];
	GNG::HSV2DVector* d_nodes;
	
	gpuErrchk( cudaMalloc((void**)&d_nodes, sizeof(GNG::HSV2DVector)) );
	gpuErrchk( cudaMemcpy((void*)d_nodes, (void*)&h_nodes, sizeof(GNG::HSV2DVector), cudaMemcpyHostToDevice) );
	
	cv::cuda::GpuMat d_connections(maxNumNodes, maxNumNodes, CV_8U);
	
	GNG::SegmentationGNG* gng = new GNG::SegmentationGNG(d_nodes, d_connections);
	*gngPointer = gng;
	
	int blockSide = (int)sqrt(dev_info.maxThreadsPerBlock());
	dim3 cthreads_blockDim(blockSide, blockSide);
	dim3 cblocks_gridDim(
		static_cast<int>(std::ceil(maxNumNodes /
		    static_cast<double>(cthreads_blockDim.x))),
		static_cast<int>(std::ceil(maxNumNodes / 
		    static_cast<double>(cthreads_blockDim.y))));

	cudaStream_t stream = cv::cuda::StreamAccessor::getStream(_stream);
	gpuNumDifferentKernel<<<cblocks_gridDim, cthreads_blockDim, 0, stream>>>(maxNumNodes, d_img, d_nodes, d_connections);

	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	//gpuErrchk( cudaMemcpy((void*)&h_diff, (void*)d_diff, sizeof(int), cudaMemcpyDeviceToHost) );

	return 0;
}
