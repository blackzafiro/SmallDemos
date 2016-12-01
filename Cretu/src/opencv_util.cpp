#include <iostream>
#include "opencv_util.h"

/**
 * Prints opencv type
 * http://stackoverflow.com/questions/10167534/how-to-find-out-what-type-of-a-mat-object-is-with-mattype-in-opencv
 * http://ninghang.blogspot.mx/2012/11/list-of-mat-type-in-opencv.html
 */
std::string type2str(int type) {
	std::string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch ( depth ) {
		case CV_8U:  r = "8U"; break;
		case CV_8S:  r = "8S"; break;
		case CV_16U: r = "16U"; break;
		case CV_16S: r = "16S"; break;
		case CV_32S: r = "32S"; break;
		case CV_32F: r = "32F"; break;
		case CV_64F: r = "64F"; break;
		default:     r = "User"; break;
	}

	r = "CV_" + r;
	r += "C";
	r += (chans+'0');

	return r;
}


/**
 * Prints codec name according to:
 * http://docs.opencv.org/trunk/d0/d61/group__cudacodec.html#ga71943a1181287609b5d649f53ce6c146
 */
std::ostream& operator<<(std::ostream& out, const cv::cudacodec::Codec value){
    const char* s = 0;
#define VAL_TO_STRING(p) case(cv::cudacodec::p): s = #p; break;
    switch(value){
        VAL_TO_STRING(MPEG1);     
        VAL_TO_STRING(MPEG2);     
        VAL_TO_STRING(MPEG4);
        VAL_TO_STRING(VC1);
        VAL_TO_STRING(H264);
        VAL_TO_STRING(JPEG);
        VAL_TO_STRING(H264_SVC);
        VAL_TO_STRING(H264_MVC);
        VAL_TO_STRING(Uncompressed_YUV420);
        VAL_TO_STRING(Uncompressed_YV12);
        VAL_TO_STRING(Uncompressed_NV12);
        VAL_TO_STRING(Uncompressed_YUYV);
        VAL_TO_STRING(Uncompressed_UYVY);
    }
#undef PROCESS_VAL
    return out << s;
}

/**
 * Checks if there is cuda compatible device available.
 */
int verifyCUDACapabilities() {
	int idev = cv::cuda::getCudaEnabledDeviceCount();
	std::cout << "This computer has " << idev << " cuda enabled device(s)." << std::endl;
	if (idev == 0) return -1;
}