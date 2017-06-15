#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <limits>


// http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __func__, __LINE__, false); }

inline void gpuAssert(cudaError_t code, const char *file, const char *func, int line, bool abort = true)
{    
    if (code != cudaSuccess) {
	fprintf(stderr, "\e[1;31mGPUassert: %s %s %s %d\n\e[1;0m", cudaGetErrorString(code), file, func, line);
	if (abort) exit(code);
    }
}

__global__ void gpuLineKernel(cv::cuda::PtrStepSz<uchar3> img,
	cv::Point* d_pt1, cv::Point* d_pt2, double m, bool isInf,
	int b, int g, int r, int alfa, int thickness)
{
    //int blockId = blockIdx.x + blockIdx.y * gridDim.x;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int min_y = d_pt1->y, max_y = d_pt2->y;
    if (min_y > max_y) {
	min_y = max_y;
	max_y = d_pt1->y;
    }
    int min_x = d_pt1->x, max_x = d_pt2->x;
    if (min_x > max_x) {
	min_x = max_x;
	max_x = d_pt1->x;
    }

    if (isInf) {
	// Vertical line
	double diff = x - d_pt1->x;
	if (y > min_y && y < max_y && sqrt(diff * diff) < thickness) {
	    img(y, x).x = b;
	    img(y, x).y = g;
	    img(y, x).z = r;
	}
    } else if (m * m < 0.0001) {
	// Horizontal line
	double diff = y - d_pt1->y;
	if (x > min_x && x < max_x && sqrt(diff * diff) < thickness) {
	    img(y, x).x = b;
	    img(y, x).y = g;
	    img(y, x).z = r;
	}
    } else {

	if (y > min_y && y < max_y && x > min_x && x < max_x) {
	    double yy = m * (x - d_pt1->x) + d_pt1->y;
	    double diff = yy - y;
	    if (sqrt(diff * diff) < thickness) {
		img(y, x).x = b;
		img(y, x).y = g;
		img(y, x).z = r;
	    }
	}
    }

}

__global__ void gpuCircleKernel(cv::cuda::PtrStepSz<uchar3> img,
	cv::Point* d_center, int radius,
	int b, int g, int r, int alfa, int thickness)
{
    //int blockId = blockIdx.x + blockIdx.y * gridDim.x;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    double dx = d_center->x - x;
    double dy = d_center->y - y;

    if (thickness < 0) {
	if (sqrt(dx * dx + dy * dy) < radius) {
	    img(y, x).x = b;
	    img(y, x).y = g;
	    img(y, x).z = r;
	}
    } else {
	if (abs(sqrt(dx * dx + dy * dy) - radius) < thickness) {
	    img(y, x).x = b;
	    img(y, x).y = g;
	    img(y, x).z = r;
	}
    }
}


__device__ double sign(double x1, double y1, double x2, double y2, double x3, double y3)
{
    return (x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3);
}


__global__ void gpuTriangleKernel(cv::cuda::PtrStepSz<uchar3> img,
	cv::Point* d_pt1, cv::Point* d_pt2, cv::Point* d_pt3,
	int b, int g, int r, int alfa, int thickness)
{
    //int blockId = blockIdx.x + blockIdx.y * gridDim.x;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    double dx = (double)x;
    double dy = (double)y;
    
    double x1 = d_pt1->x;
    double x2 = d_pt2->x;
    double x3 = d_pt3->x;
    double y1 = d_pt1->y;
    double y2 = d_pt2->y;
    double y3 = d_pt3->y;

    bool b1, b2, b3;
    b1 = sign(dx, dy, x1, y1, x2, y2) < 0.0f;
    b2 = sign(dx, dy, x2, y2, x3, y3) < 0.0f;
    b3 = sign(dx, dy, x3, y3, x1, y1) < 0.0f;

    if ((b1 == b2) && (b2 == b3)) {
	img(y, x).x = b;
	img(y, x).y = g;
	img(y, x).z = r;
    }
}

/*
 * Draw line.  Like cv::line, but in gpumat.
 */
int gpuLine(cv::InputArray _img, cv::Point pt1, cv::Point pt2, const cv::Scalar& color, int thickness = 1) {
    const cv::cuda::GpuMat img = _img.getGpuMat();

    dim3 cthreads_blockDim(32, 32);
    dim3 cblocks_gridDim(
	    static_cast<int>(std::ceil(img.size().width /
	    static_cast<double>(cthreads_blockDim.x))),
	    static_cast<int>(std::ceil(img.size().height /
	    static_cast<double>(cthreads_blockDim.y))));

    cv::Point *d_pt1;
    cv::Point *d_pt2;
    double m;
    bool isInf = false;

    if ((pt2.x - pt1.x) * (pt2.x - pt1.x) > 0.0001) {
	m = (pt2.y - pt1.y) / (pt2.x - pt1.x);
    } else {
	m = std::numeric_limits<double>::infinity();
	isInf = true;
    }


    gpuErrchk(cudaMalloc((void**) &d_pt1, sizeof (cv::Point)));
    gpuErrchk(cudaMalloc((void**) &d_pt2, sizeof (cv::Point)));

    gpuErrchk(cudaMemcpy((void*) d_pt1, (void*) &pt1, sizeof (cv::Point), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy((void*) d_pt2, (void*) &pt2, sizeof (cv::Point), cudaMemcpyHostToDevice));

    cv::cuda::Stream _stream = cv::cuda::Stream();
    cudaStream_t stream = cv::cuda::StreamAccessor::getStream(_stream);
    gpuLineKernel << <cblocks_gridDim, cthreads_blockDim, 0, stream>>>(img, d_pt1, d_pt2, m, isInf, color[0], color[1], color[2], color[3], thickness);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    cudaFree(d_pt1);
    cudaFree(d_pt2);

    return 0;
}

/*
 * Draw circle.  Like cv::line, but in gpumat.
 */
int gpuCircle(cv::InputArray _img, cv::Point center, int radius, const cv::Scalar& color, int thickness = 1) {
    const cv::cuda::GpuMat img = _img.getGpuMat();

    dim3 cthreads_blockDim(32, 32);
    dim3 cblocks_gridDim(
	    static_cast<int>(std::ceil(img.size().width /
	    static_cast<double>(cthreads_blockDim.x))),
	    static_cast<int>(std::ceil(img.size().height /
	    static_cast<double>(cthreads_blockDim.y))));

    cv::Point *d_center;

    gpuErrchk(cudaMalloc((void**) &d_center, sizeof (cv::Point)));

    gpuErrchk(cudaMemcpy((void*) d_center, (void*) &center, sizeof (cv::Point), cudaMemcpyHostToDevice));

    cv::cuda::Stream _stream = cv::cuda::Stream();
    cudaStream_t stream = cv::cuda::StreamAccessor::getStream(_stream);
    gpuCircleKernel << <cblocks_gridDim, cthreads_blockDim, 0, stream>>>(img, d_center, radius, color[0], color[1], color[2], color[3], thickness);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    cudaFree(d_center);

    return 0;
}

/**
 * Draw triangle.
 */
int gpuFillTriangle(cv::InputArray _img,
	cv::Point pt1, cv::Point pt2, cv::Point pt3,
	const cv::Scalar& color, int thickness = 1) {
    const cv::cuda::GpuMat img = _img.getGpuMat();

    dim3 cthreads_blockDim(32, 32);
    dim3 cblocks_gridDim(
	    static_cast<int>(std::ceil(img.size().width /
	    static_cast<double>(cthreads_blockDim.x))),
	    static_cast<int>(std::ceil(img.size().height /
	    static_cast<double>(cthreads_blockDim.y))));

    cv::Point *d_pt1;
    cv::Point *d_pt2;
    cv::Point *d_pt3;

    gpuErrchk(cudaMalloc((void**) &d_pt1, sizeof (cv::Point)));
    gpuErrchk(cudaMalloc((void**) &d_pt2, sizeof (cv::Point)));
    gpuErrchk(cudaMalloc((void**) &d_pt3, sizeof (cv::Point)));

    gpuErrchk(cudaMemcpy((void*) d_pt1, (void*) &pt1, sizeof (cv::Point), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy((void*) d_pt2, (void*) &pt2, sizeof (cv::Point), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy((void*) d_pt3, (void*) &pt3, sizeof (cv::Point), cudaMemcpyHostToDevice));

    cv::cuda::Stream _stream = cv::cuda::Stream();
    cudaStream_t stream = cv::cuda::StreamAccessor::getStream(_stream);
    gpuTriangleKernel << <cblocks_gridDim, cthreads_blockDim, 0, stream>>>(img, d_pt1, d_pt2, d_pt3, color[0], color[1], color[2], color[3], thickness);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    cudaFree(d_pt1);
    cudaFree(d_pt2);
    cudaFree(d_pt3);

    return 0;
}