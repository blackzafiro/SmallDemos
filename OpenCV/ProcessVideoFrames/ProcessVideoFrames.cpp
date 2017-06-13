#include <iostream>
#include <string>

#include "opencv2/opencv_modules.hpp"

#if defined(HAVE_OPENCV_CUDACODEC)

#include <string>
#include <vector>
#include <algorithm>
#include <numeric>

#include <opencv2/core.hpp>
#include <opencv2/core/opengl.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/highgui.hpp>

// Example of algorithm already in OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>

// This function is defined in .cu file
int gpuNumDifferent(cv::InputArray _img1,
                     cv::InputArray _img2);

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

int main(int argc, const char* argv[])
{
	// Test if cuda device is present

	int idev = cv::cuda::getCudaEnabledDeviceCount();
	std::cout << "This computer has " << idev << " cuda enabled device(s)." << std::endl;
	if (idev == 0) return -1;

	if (argc != 2) {
		std::cerr << "Use: VideoReader <video_file>" << std::endl;
		return -1;
	}

	const std::string fname(argv[1]);

	cv::namedWindow("GPU", cv::WINDOW_OPENGL);  cv::moveWindow("GPU", 10, 50);
	cv::namedWindow("Mod", cv::WINDOW_OPENGL);  cv::moveWindow("Mod", 500, 50);
	cv::cuda::setGlDevice();

	// Ask if device is compatible
	cv::cuda::DeviceInfo dev_info = cv::cuda::DeviceInfo();
	bool bglcomp = dev_info.isCompatible();
	std::cout << "This computer's device compatibility with GlDevice is " << bglcomp << std::endl;
	if (!bglcomp) return -1;

	cv::cuda::GpuMat d_frame;
	cv::Ptr<cv::cudacodec::VideoReader> d_reader = cv::cudacodec::createVideoReader(fname);
	cv::cuda::GpuMat d_dst;

	cv::cudacodec::FormatInfo formatInfo = d_reader->format();
	std::cout << formatInfo.width << "x" << formatInfo.height << " codec:" << formatInfo.codec << std::endl;


	int nframe = 0;
	int ndiff = 0;
	for (;;)
	{
		// Read frame in original video
		if (!d_reader->nextFrame(d_frame))
			break;
		cv::imshow("GPU", d_frame);

		// Flip x and y with algorithm from cudaarithm.hpp
		cv::cuda::flip(d_frame, d_dst, -1);
		cv::imshow("Mod", d_dst);

		if(d_frame.type() == CV_8UC4)
		{
			// Count different pixels between images with custom function
			ndiff = gpuNumDifferent(d_frame, d_dst);

			std::cout << "Frame " << nframe++ << ": there are " << ndiff << " different "
		              << type2str(d_frame.type()) << " " << d_frame.size() << " pixels" << std::endl;
		}
		else
		{
			std::cout << "Custom kernel function only works with RGBA videos decoded to CV_8UC4" << std::endl;
		}


        if (cv::waitKey(3) == 'q')
			break;

	}

	return 0;
}


#else

int main()
{
    std::cout << "OpenCV was built without CUDA Video decoding support\n" << std::endl;
    return 0;
}

#endif
