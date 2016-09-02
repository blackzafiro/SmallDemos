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
#include <opencv2/cudaimgproc.hpp> 

// Utility functions
#include "opencv_util.h"

// This function is defined in .cu file
int gpuSegment(cv::InputArray _img1,
                     cv::OutputArray _img2,
                     cv::cuda::Stream _stream);

int main(int argc, const char* argv[])
{
	// Test if cuda device is present

	int idev = cv::cuda::getCudaEnabledDeviceCount();
	std::cout << "This computer has " << idev << " cuda enabled device(s)." << std::endl;
	if (idev == 0) return -1;

	if (argc != 2) {
		std::cerr << "Use: ProcessVideoFrames <video_file>" << std::endl;
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

	// >> Read initial frame
	int nframe = 0;
	if (!d_reader->nextFrame(d_frame)) {
		std::cerr << "Could not read first frame" << std::endl;
		exit(1);
	}
	// >> Transform to HSV
	cv::cuda::GpuMat d_hsv; // = cv::cuda::GpuMat(d_frame.size(), cv::CV_8UC3);
	//d_hsv.upload(d_frame);
	cv::cuda::cvtColor(d_frame, d_hsv, cv::COLOR_BGR2HSV);
	//cv::imshow("Mod", d_hsv);

	//cv::waitKey(0);
	
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
			ndiff = gpuSegment(d_frame, d_dst, cv::cuda::Stream());

			std::cout << "Frame " << nframe++ << ": there are " << ndiff << " different "
		              << type2str(d_frame.type()) << " " << d_frame.size() << " pixels" << std::endl;
		}
		else
		{
			std::cout << "Custom kernel function only works with RGBA videos decoded to CV_8UC4" << std::endl;
		}

		if (cv::waitKey(3) > 0)
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
