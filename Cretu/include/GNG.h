#ifndef GNG_H
#define GNG_H

#include <opencv2/cudaimgproc.hpp>

namespace GNG {

	class HSV2DVector {
	};

	class SegmentationGNG {
	private:
		HSV2DVector* dp_nodes;
		cv::cuda::GpuMat& dr_connections;
		
	public:
		/** Constructor. */
		SegmentationGNG(HSV2DVector* d_nodes, cv::cuda::GpuMat &connections);
		~SegmentationGNG();
	};

	/**
	 * Uses Fritzke GNG to segment figure and backgroud from given image.
	 * Returns a negative number if there was an error.
	 */
	//int segmentFrame(cv::cuda::GpuMat&);

}

#endif
