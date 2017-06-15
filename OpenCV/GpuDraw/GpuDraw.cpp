#include <iostream>

#include "opencv2/opencv_modules.hpp"

#if defined(HAVE_OPENCV_CUDACODEC)

#include <opencv2/core.hpp>
#include <opencv2/core/opengl.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/highgui.hpp>

#include <opencv2/opencv.hpp>

int gpuLine(cv::InputArray _img1, cv::Point pt1, cv::Point pt2, const cv::Scalar& color, int thickness = 1);
int gpuCircle(cv::InputArray _img, cv::Point center, int radius, const cv::Scalar& color, int thickness = 1);
int gpuFillTriangle(cv::InputArray _img, cv::Point pt1, cv::Point pt2, cv::Point pt3, const cv::Scalar& color, int thickness = 1);

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
	
	gpuFillTriangle(d_img, cv::Point(150,50), cv::Point(450,250), cv::Point(250,250), cv::Scalar(25,67,100));
	gpuFillTriangle(d_img, cv::Point(150,250), cv::Point(250,450), cv::Point(450,450), cv::Scalar(125,167,255));
			
	gpuLine(d_img, cv::Point(250,250), cv::Point(250,300), cv::Scalar(25,67,100), 5);
	gpuLine(d_img, cv::Point(50,25), cv::Point(250,300), cv::Scalar(255,0,0));
	gpuLine(d_img, cv::Point(50,25), cv::Point(250,600), cv::Scalar(0,255,0), 2);
    gpuLine(d_img, cv::Point(500,25), cv::Point(2,600), cv::Scalar(0,0,255), 8);
    gpuLine(d_img, cv::Point(400,300), cv::Point(25,300), cv::Scalar(250,67,100), 5);

    gpuCircle(d_img, cv::Point(400,350), 30, cv::Scalar(250,80,150), -3);
    gpuCircle(d_img, cv::Point(100,300), 30, cv::Scalar(250,80,150), 6);
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
