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


 __global__ void gpuLineKernel(cv::cuda::PtrStepSz<uchar3> img,
								cv::Point* d_pt1, cv::Point* d_pt2, double m,
								cv::Scalar* d_color)
{
	int blockId = blockIdx.x + blockIdx.y * gridDim.x; 

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	img(y, x).x = 255;
}



/*
 * Draw line.  Like cv::line, but in gpumat.
 */
int gpuLine(cv::InputArray _img, cv::Point pt1, cv::Point pt2, const cv::Scalar& color)
{
	const cv::cuda::GpuMat img = _img.getGpuMat();

	dim3 cthreads_blockDim(32, 32);
	dim3 cblocks_gridDim(
		static_cast<int>(std::ceil(img.size().width /
		    static_cast<double>(cthreads_blockDim.x))),
		static_cast<int>(std::ceil(img.size().height / 
		    static_cast<double>(cthreads_blockDim.y))));

	cv::Point *d_pt1;
	cv::Point *d_pt2;
	cv::Scalar *d_color;

	gpuErrchk( cudaMalloc((void**)&d_pt1, sizeof(cv::Point)) );
	gpuErrchk( cudaMalloc((void**)&d_pt2, sizeof(cv::Point)) );
	gpuErrchk( cudaMalloc((void**)&d_color, sizeof(cv::Scalar)) );

	gpuErrchk( cudaMemcpy((void*)d_pt1, (void*)&pt1, sizeof(cv::Point), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy((void*)d_pt2, (void*)&pt2, sizeof(cv::Point), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy((void*)d_color, (void*)&color, sizeof(cv::Scalar), cudaMemcpyHostToDevice) );

	cv::cuda::Stream _stream = cv::cuda::Stream();
	cudaStream_t stream = cv::cuda::StreamAccessor::getStream(_stream);
	gpuLineKernel<<<cblocks_gridDim, cthreads_blockDim, 0, stream>>>(img, d_pt1, d_pt2, d_color);

	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	cudaFree(d_pt1);
	cudaFree(d_pt2);
	cudaFree(d_color);

	return 0;
}