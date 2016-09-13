#ifndef GNG_H
#define GNG_H

#include <opencv2/cudaimgproc.hpp> 

/**
 * Uses Fritzke GNG to segment figure and backgroud from given image.
 * Returns a negative number if there was an error.
 */
int segmentFrame(cv::cuda::GpuMat&);

#endif