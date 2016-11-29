#include <iostream>

#include "opencv2/opencv_modules.hpp"

#if defined(HAVE_OPENCV_CUDACODEC)

#include <opencv2/core.hpp>
#include <opencv2/core/opengl.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/highgui.hpp>

#include <opencv2/opencv.hpp>

int gpuLine(cv::InputArray _img1, cv::Point pt1, cv::Point pt2, const cv::Scalar& color);

int main(int argc, const char* argv[])
{
	// Test if cuda device is present

	int idev = cv::cuda::getCudaEnabledDeviceCount();
	std::cout << "This computer has " << idev << " cuda enabled device(s)." << std::endl;
    if (idev == 0) return -1;
    
    cv::namedWindow("GPU", cv::WINDOW_OPENGL);  cv::moveWindow("GPU", 10, 50);
    cv::cuda::setGlDevice();

    // Ask if device is compatible
	cv::cuda::DeviceInfo dev_info = cv::cuda::DeviceInfo();
	bool bglcomp = dev_info.isCompatible();
	std::cout << "This computer's device compatibility with GlDevice is " << bglcomp << std::endl;
	if (!bglcomp) return -1;

	cv::cuda::GpuMat d_img(500, 500, CV_8UC3);
	d_img.setTo(cv::Scalar::all(0));
	//cv::circle(d_img, cv::Point(250,250), 50, cv::Scalar(25,67,100));
	//cv::line(d_img, cv::Point(250,250), cv::Point(250,300), cv::Scalar(25,67,100));
	gpuLine(d_img, cv::Point(250,250), cv::Point(250,300), cv::Scalar(25,67,100));

    cv::imshow("GPU", d_img);

	cvWaitKey(0);

    return 0;
}

#else

int main()
{
    std::cout << "OpenCV was built without CUDA Video decoding support\n" << std::endl;
    return 0;
}

#endif
