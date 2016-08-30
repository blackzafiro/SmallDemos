#ifndef OPENCV_UTIL_H
#define OPENCV_UTIL_H

#include <string>

#include <opencv2/core.hpp>
#include <opencv2/cudacodec.hpp>

/**
 * Prints opencv type
 * http://stackoverflow.com/questions/10167534/how-to-find-out-what-type-of-a-mat-object-is-with-mattype-in-opencv
 * http://ninghang.blogspot.mx/2012/11/list-of-mat-type-in-opencv.html
 */
std::string type2str(int type);

/**
 * Prints codec name according to:
 * http://docs.opencv.org/trunk/d0/d61/group__cudacodec.html#ga71943a1181287609b5d649f53ce6c146
 */
std::ostream& operator<<(std::ostream& out, const cv::cudacodec::Codec value);

#endif