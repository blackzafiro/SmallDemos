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
#include "GNG.h"

/**
 * Calculates a region of interest centered on the image, with the maximum size
 * for threading.
 */
cv::Rect calculateROI(int imageWidth, int imageHeight, int devWidth, int devHeight) {
	int x(0), y(0), width(imageWidth), height(imageHeight);
	if (imageWidth > devWidth) {
		x = (imageWidth - devWidth) / 2;
		width = devWidth;
	}
	if (imageHeight > devHeight) {
		y = (imageHeight - devHeight) / 2;
		width = devHeight;
	}
	return cv::Rect(x, y, width, height);
}

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

	// Ask if device is compatible
	cv::cuda::DeviceInfo dev_info = cv::cuda::DeviceInfo();
	bool bglcomp = dev_info.isCompatible();
	std::cout << "This computer's device compatibility with GlDevice is " << bglcomp << std::endl;
	if (!bglcomp) return -1;

	// Ask dimensions of blocks and threads
	int numKernels = dev_info.concurrentKernels();
	std::cout << "There can be " << numKernels << " concurrentKernel(s)" << std::endl;
	cv::Vec<int, 3> gridSize = dev_info.maxGridSize();
	std::cout << "Max grid size  " << gridSize << std::endl; // 960: Max grid size  [2147483647, 65535, 65535]
	cv::Vec<int, 3> blockSize = dev_info.maxThreadsDim();
	std::cout << "Max threads per block " << blockSize << std::endl; // 960: Max threads per block [1024, 1024, 64]

	if (argc != 2) {
		std::cerr << "Use: ProcessVideoFrames <video_file>" << std::endl;
		return -1;
	}

	const std::string fname(argv[1]);

	cv::namedWindow("GPU", cv::WINDOW_OPENGL);  cv::moveWindow("GPU", 10, 50);
	cv::namedWindow("Mod", cv::WINDOW_OPENGL);  cv::moveWindow("Mod", 500, 50);
	cv::cuda::setGlDevice();

	cv::cuda::GpuMat d_frame;
	cv::Ptr<cv::cudacodec::VideoReader> d_reader = cv::cudacodec::createVideoReader(fname);
	cv::cuda::GpuMat d_dst;

	cv::cudacodec::FormatInfo formatInfo = d_reader->format();
	std::cout << formatInfo.width << "x" << formatInfo.height << " codec:" << formatInfo.codec << std::endl;

	int imageWidth = formatInfo.width;
	int imageHeight = formatInfo.height;

	// >> Read initial frame
	int nframe = 0;
	if (!d_reader->nextFrame(d_frame)) {
		std::cerr << "Could not read first frame" << std::endl;
		exit(1);
	}
	// Get region of interest, according to device capabilites
	cv::Rect roi = calculateROI(imageWidth, imageHeight, blockSize[0], blockSize[1]);
	cv::cuda::GpuMat d_roi = cv::cuda::GpuMat(d_frame, roi);
	//cv::rectangle(d_roi, roi.tl(), roi.br(), cv::Scalar(255), 1, 8, 0);

	// >> Transform to HSV
	cv::cuda::GpuMat d_hsv;
	cv::cuda::cvtColor(d_roi, d_hsv, cv::COLOR_BGR2HSV);
	cv::imshow("GPU", d_frame);
	cv::imshow("Mod", d_hsv);

	if(segmentFrame(d_hsv) < 0) {
		std::cerr << "Segmentation of first frame failed." << std::endl;
		exit(1);
	}

	cv::waitKey(0);
	/*
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
	*/
	return 0;
}


#else

int main()
{
    std::cout << "OpenCV was built without CUDA Video decoding support\n" << std::endl;
    return 0;
}

#endif
