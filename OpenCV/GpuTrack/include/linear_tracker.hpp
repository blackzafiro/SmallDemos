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
	LinearSplineTracker(std::vector<cv::Point>& border_clue, int max_distance, int min_distance, int min_angle, int rings);
	
	/**
	 * Adjust linear spline to edges and save in history.
	 * @param d_img  Image with edges of the object to track.
	 * @param d_draw Countour will be drawn in this 4 channel image.
	 */
	void track(cv::cuda::GpuMat d_img, cv::cuda::GpuMat d_edges, cv::cuda::GpuMat d_draw);
	
	/**
	 * Writes the history of this tracking session to <code>o</code>.
	 * @param o Output stream where data will be written.
	 * @return Same stream.
	 */
	std::ostream& write(std::ostream &o) const;
	
	~LinearSplineTracker();
	
private:
	LinearSplineTrackerImpl *impl_;
};

#endif /* LINEAR_TRACKER_H */

