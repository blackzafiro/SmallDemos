#include "GNG.h"
#include <cuda_runtime.h>

/** Constructor. */
GNG::SegmentationGNG::SegmentationGNG(HSV2DVector* d_nodes, cv::cuda::GpuMat &connections) :
	dp_nodes(d_nodes), 
	dr_connections(connections) {}

GNG::SegmentationGNG::~SegmentationGNG() {
	if(dp_nodes) cudaFree(dp_nodes);
}
