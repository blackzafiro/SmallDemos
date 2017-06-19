#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __func__, __LINE__, false); }

inline void gpuAssert(cudaError_t code, const char *file, const char *func, int line, bool abort = true)
{    
    if (code != cudaSuccess) {
	fprintf(stderr, "\e[1;31mGPUassert: %s %s %s %d\n\e[1;0m", cudaGetErrorString(code), file, func, line);
	if (abort) exit(code);
    }
}

class ControlPoint
{
private:
	int x;
	int y;
	int left_neighbour_index;
	int right_neighbour_index;
};

class ControlPointHistory
{
private:
	ControlPoint* control_point;
};

class ContourHistory
{
	ControlPointHistory* control_point_history;
};

class LinearSplineTrackerImpl
{
	
};