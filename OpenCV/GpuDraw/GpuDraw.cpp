#include <iostream>

#include "opencv2/opencv_modules.hpp"

#if defined(HAVE_OPENCV_CUDACODEC)

#include <opencv2/core.hpp>
#include <opencv2/core/opengl.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/highgui.hpp>

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

    cv::imshow("GPU", d_img);

    return 0;
}

#else

int main()
{
    std::cout << "OpenCV was built without CUDA Video decoding support\n" << std::endl;
    return 0;
}

#endif
