#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

#include "linear_tracker.hpp"
#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>


#define CPH_INITIAL_LENGHT		20	// Initial capacity of array of control points.
#define CONTOUR_INITIAL_LENGTH	100	// Initial estimate of the number of control points that will appear.

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __func__, __LINE__, false); }

inline void gpuAssert(cudaError_t code, const char *file, const char *func, int line, bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "\e[1;31mGPUassert: %s %s %s %d\n\e[1;0m", cudaGetErrorString(code), file, func, line);
		if (abort) exit(code);
	}
}

class ControlPoint{
private:
	int _x;
	int _y;
	int _prev_neighbour_index;
	int _next_neighbour_index;

public:	
	__host__ __device__
	ControlPoint() :
		_x(-1), _y(-1), _prev_neighbour_index(-1), _next_neighbour_index(-1) {}

	/*__host__ __device__
	ControlPoint(int x, int y, int prev_neighbour_index, int next_neighbour_index) :
		x(x), y(y), prev_neighbour_index(prev_neighbour_index), next_neighbour_index(next_neighbour_index) {}*/

	__device__
	int x() { return _x; }

	__device__
	int y() { return _y; }

	__device__
	int prev_neighbour_index() { return _prev_neighbour_index; }

	__device__
	int next_neighbour_index() { return _next_neighbour_index; }

	__device__
	ControlPoint(const ControlPoint& other) :
		_x(other._x), _y(other._y),
		_prev_neighbour_index(other._prev_neighbour_index),
		_next_neighbour_index(other._next_neighbour_index)
	{
	}
	
	__host__ __device__
	void set(int x, int y, int prev_neighbour_index, int next_neighbour_index)
	{
		this->_x = x;
		this->_y = y;
		this->_prev_neighbour_index = prev_neighbour_index;
		this->_next_neighbour_index = next_neighbour_index;
	}
	
	/*
	__device__
	ControlPoint& operator=(const ControlPoint& other)
	{
		if (this != &other)		// self-assignment check expected
		{ 
			this.x = other.x;
			this.y = other.y;
			this.left_neighbour_index = other.left_neighbour_index;
			this.right_neighbour_index = other.right_neighbour_index;
		}
		return *this;
	}
	*/
	
	__host__ __device__
	~ControlPoint() {}
};

/** Dinamic array of control points each object shall be accessed only by one GPU. */
class ControlPointHistory{
private :
	int birth_time;
	int death_time;

	ControlPoint* control_point;	// Pointer to array of control points
	int end;						// Last element in array
	int length;						// Size of array

	/**
	 * Doubles the capacity of the array.
	 * Returns a negative number if memory operations failed.
	 */
	__device__
	int resize()
	{
		ControlPoint* new_array;
		int new_length = length * 2;
		cudaError_t code = cudaMalloc((void**) &new_array, sizeof(ControlPoint) * new_length);
		if (code != cudaSuccess)
		{
			return -1;
		}
		memcpy ( (void*)new_array, (void*)control_point, sizeof(ControlPoint) * length);
		if (code != cudaSuccess)
		{
			return -2;
		}

		cudaFree(control_point);
		control_point = new_array;
		length = new_length;

		return 0;
	}

public:
	__device__
	ControlPointHistory(int birth_time) :  birth_time(birth_time), death_time(-1), end(0), length(CPH_INITIAL_LENGHT)
	{
		cudaError_t code = cudaMalloc((void**) &control_point, sizeof(ControlPoint) * CPH_INITIAL_LENGHT);
		if (code != cudaSuccess)
		{
			length = -1;
			return;
		}
	}
	
	__device__
	void queue(ControlPoint &cpoint)
	{
		if (end == length) resize();
		control_point[end].set(cpoint.x(), cpoint.y(), cpoint.prev_neighbour_index(), cpoint.next_neighbour_index());
		end++;
	}
	
	__device__
	bool isAlive()
	{
		return death_time == -1;
	}
	
	__device__
	void remove()
	{
		death_time = end - 1;
	}
	
	__device__
	~ControlPointHistory()
	{
		cudaFree(control_point);
	}
};

/**
 * Class in charge of creating the history from the host and granting access to
 * it from the device.
 */
class ContourHistory{
private:
	ControlPointHistory** d_control_point_history;	// Array of pointers to created histories.
	unsigned int end;
	unsigned int capacity;

	int time_step;

	__device__
	void resize()
	{
		ControlPointHistory** new_array;
		int new_capacity = capacity * 2;
		gpuErrchk( cudaMalloc((void**) &new_array, sizeof(ControlPointHistory*) * new_capacity) );
		gpuErrchk( cudaMemcpy ( (void*)new_array, (void*)d_control_point_history, sizeof(ControlPointHistory*) * capacity,  cudaMemcpyDeviceToDevice) );

		cudaFree(d_control_point_history);
		d_control_point_history = new_array;
		capacity = new_capacity;
	}

public:

	/**
	 * Constructs dynamic array of pointers to control point histories.
	 */
	__device__
	ContourHistory(int initial_length) : end(0) , time_step(0)
	{
		int num_elems = (initial_length > CONTOUR_INITIAL_LENGTH) ? initial_length : CONTOUR_INITIAL_LENGTH;

		cudaError_t code = cudaMalloc((void**) &d_control_point_history, sizeof (ControlPointHistory*)  * num_elems);
		if (code != cudaSuccess) {
			num_elems = -1;
			return;
		}

		capacity = num_elems;
	}

	/**
	 * To be called by the last GPU when values are written in parallel.
	 * It sets the end of the list to the length of the TDA (Do not confusse with the capacity of the array implementation.
	 */
	__device__
	void setLength(int length)
	{
		this->end = length;
	}

	/**
	 * Read number of elements in list.
	 */
	__device__
	unsigned int getLength()
	{
		return end;
	}

	/**
	 * Creates control point history, mean to be used by the initialization kernel.
	 * @param cpoint data of the control point.
	 * @param index position where the history will be created, it must be smaller than the capacity of the list.
	 */
	__device__
	void createControlPointHistory(ControlPoint &cpoint, int index)
	{
		//ControlPoint* h_point = new ControlPoint(cpoint);
		//d_control_point_history[index] = d_control_point_history;
		ControlPointHistory* d_cph = new ControlPointHistory(this->time_step);
		d_cph->queue(cpoint);
		d_control_point_history[index] = d_cph;
	}

	__device__
	~ContourHistory()
	{
		for(int i = 0; i < end; i++)
		{
			cudaFree(d_control_point_history[i]);
		}
		cudaFree(d_control_point_history);
	}
};

// ------ Kernel ------

//__global__
//int devMemcpy(void* dst, void* src, size_t count)

/**
 * Creates the contour history object in device memory.
 * Only one gpu must be requested in kernell call.
 */
__global__
void createTracker(ContourHistory* d_contour_history, int minimum_length)
{
	d_contour_history = new ContourHistory(minimum_length);
}

/**
 * Initialices contour_history and saves the pointer to it in d_contour_history.
 * To be called by as many GPUs as points in the initial suggested contour in one single block.
 */
__global__
void initTracker(ContourHistory* d_contour_history, unsigned int* d_contour_history_length, ControlPoint* d_point_array, int length)
{
	int index = blockIdx.x *blockDim.x + threadIdx.x;

	d_contour_history->createControlPointHistory(d_point_array[index], index);

	__syncthreads();

	if(index == length - 1)
	{
		// Update number of control points histories being recorded
		d_contour_history->setLength(length);
		// Update external length information
		*d_contour_history_length = d_contour_history->getLength();
	}
}

/**
 * Free device memory with contour history object.
 */
__global__
void destroyTracker(ContourHistory* d_contour_history)
{
	delete d_contour_history;
}

/**
 * The kernel must have only one dimension.
 * Each gpu is in charge of a control point.
 */
__global__
void trackKernell(cv::cuda::PtrStepSz<uchar3> d_img, cv::cuda::PtrStepSz<uchar1> d_edges, cv::cuda::PtrStepSz<uchar3> d_draw, unsigned int* d_contour_history_length)
{
	int index = blockIdx.x *blockDim.x + threadIdx.x;
	int c_length = *d_contour_history_length;

	if(index == c_length) {
		*d_contour_history_length = index;
	}
}


// ------ Device variables ------

//__device__ unsigned int d_contour_history_length;

// ------ Tracker classes ------


class LinearSplineTrackerImpl{
private:
	// +++ Properties
	int max_distance, min_distance, min_angle;
	int rings;		// The maximum distance to look for an edge in pixels
	
	// +++ Auxiliary variables
	int _max_threads;

	unsigned int _contour_history_length;
	unsigned int* d_contour_history_length;

	ContourHistory* d_contour_history;

	/**
	 * Add the points in border_clue to the contour history tracking structure.
	 */
	void initBorder(std::vector<cv::Point>& border_clue)
	{
		const int length = border_clue.size();

		std::cout << "Creating history of control points..." << std::endl;

		gpuErrchk( cudaMalloc((void **) &d_contour_history, sizeof(ContourHistory)) );

		createTracker <<< 1, 1 >>> (d_contour_history, length);
		gpuErrchk( cudaPeekAtLastError() );


		std::cout << "Number of control points to add for initialization: \t" << length << std::endl;

		// Creating control points in host memory
		ControlPoint h_cps[length];
		for (int i = 0; i < length; i++)
		{
			cv::Point& cv_point = border_clue[i];
			h_cps[i].set(cv_point.x, cv_point.y,
							(i - 1 < 0) ? length - 1 : i - 1,
							(i + 1) % length
			);
		}

		// Trasfering to device memory
		ControlPoint* d_point_array;
		std::cout << sizeof(ControlPoint) << " vs " << sizeof(int) << std::endl;
		gpuErrchk( cudaMalloc((void **) &d_point_array, sizeof(ControlPoint) * length) );
		gpuErrchk( cudaMemcpy(d_point_array, h_cps, sizeof(ControlPoint) * length, cudaMemcpyHostToDevice) );

		std::cout << "Number of control points in history before [init]\t" << _contour_history_length << std::endl;

		// Add control points to contour_history

		initTracker <<< 1, length >>> (d_contour_history, d_contour_history_length, d_point_array, length);
		gpuErrchk( cudaPeekAtLastError() );

		// update length of tracker
		gpuErrchk( cudaMemcpy((void*) &_contour_history_length, (const void*) d_contour_history_length, sizeof(int), cudaMemcpyDeviceToHost) );
		gpuErrchk( cudaDeviceSynchronize() );

		// free device temporary storage of control points.
		cudaFree(d_point_array);
		std::cout << "Number of control points in history after [init]\t" << _contour_history_length << std::endl;
	}

public:
	LinearSplineTrackerImpl(std::vector<cv::Point>& border_clue, int max_distance, int min_distance, int min_angle, int rings) :
	max_distance(max_distance), min_distance(min_distance), min_angle(min_angle), rings(rings), _contour_history_length (0)
	{
		struct cudaDeviceProp properties;
		cudaGetDeviceProperties(&properties, 0);	// Assuming device 0:
		_max_threads = properties.maxThreadsPerMultiProcessor;

		gpuErrchk( cudaMalloc((void **) &d_contour_history_length, sizeof(unsigned int)) );
		gpuErrchk( cudaMalloc((void **) &d_contour_history, sizeof(ContourHistory*)) );

		initBorder(border_clue);
	}

	void track(cv::cuda::GpuMat d_img, cv::cuda::GpuMat d_edges, cv::cuda::GpuMat d_draw)
	{
		std::cout << "Number of control points in history before\t" << _contour_history_length << std::endl;

		gpuErrchk( cudaMemcpy((void*) d_contour_history_length, (const void*)&_contour_history_length, sizeof(unsigned int), cudaMemcpyHostToDevice) );

		if (_contour_history_length > 0 &&  _contour_history_length <= _max_threads)
		{
			trackKernell <<< 1, _contour_history_length >>> (d_img, d_edges, d_draw, d_contour_history_length);
		}

		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaMemcpy((void*) &_contour_history_length, (const void*) d_contour_history_length, sizeof(int), cudaMemcpyDeviceToHost) );
		gpuErrchk( cudaDeviceSynchronize() );

		std::cout << "Number of control points in history after\t" << _contour_history_length << std::endl;
	}

	~LinearSplineTrackerImpl()
	{
		destroyTracker <<< 1, 1 >>> (d_contour_history);
		cudaFree(d_contour_history);
		cudaFree(d_contour_history_length);
	}

};

LinearSplineTracker::LinearSplineTracker(std::vector<cv::Point>& border_clue, int max_distance, int min_distance, int min_angle, int rings)
{
	this->impl_ = new LinearSplineTrackerImpl(border_clue, max_distance, min_distance, min_angle, rings);
}

void LinearSplineTracker::track(cv::cuda::GpuMat d_img, cv::cuda::GpuMat d_edges, cv::cuda::GpuMat d_draw)
{
	this->impl_->track(d_img, d_edges, d_draw);
}

LinearSplineTracker::~LinearSplineTracker()
{
	delete (this->impl_);
}

std::ostream& LinearSplineTracker::write(std::ostream& o) const
{
	return o;
}

