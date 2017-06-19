/* 
 * File:   linear_tracker.h
 * Author: blackzafiro
 *
 * Created on 19 de junio de 2017, 04:00 PM
 */

#ifndef LINEAR_TRACKER_H
#define LINEAR_TRACKER_H

#include <opencv2/core/cuda.hpp>

class LinearSplineTrackerImpl;

class LinearSplineTracker
{
public:
	
	/**
	 * Initialize linear snake
	 */
	LinearSplineTracker(int maxDistance, int minDistance, int minAngle, int rings) {};
	
	/**
	 * Adjust linear spline to edges and save in history.
	 * @param d_img  Image with edges of the object to track.
	 * @param d_draw Countour will be drawn in this 4 channel image.
	 */
	void track(cv::cuda::GpuMat d_img, cv::cuda::GpuMat d_draw);
	
	/**
	 * Writes the history of this tracking session to <code>o</code>.
	 * @param o Output stream where data will be written.
	 * @return Same stream.
	 */
	std::ostream& write(std::ostream &o) const;
	
private:
	LinearSplineTrackerImpl *impl_;
	
	int maxDistance, minDistance, minAngle;
	int rings;		// The maximum distance to look for an edge in pixels
};

#endif /* LINEAR_TRACKER_H */

