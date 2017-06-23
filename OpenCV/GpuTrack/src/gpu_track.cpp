#define DEBUG_SHOW

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

#include "linear_tracker.hpp"

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

struct MedianParams
{
	int window_size;
	int partition;
} median_params;

struct SobelParams
{
	int srcType;
	int dstType;
	int dx;				// Derivative order in respect of x.
	int dy;				// Derivative order in respect of y.
	int ksize;		// Size of the extended Sobel kernel. Possible values are 1, 3, 5 or 7.
	double scale;		// Optional scale factor for the computed derivative values.
} sobel_params;

struct CannyParams
{
	double low_thresh;
	double high_thresh;
	int apperture_size;
} canny_params;

struct LinearSnakeParams
{
	std::vector<cv::Point> border_clue;
	int max_distance;
	int min_distance;
	int min_angle;
	int rings;
} lsnake_params;

struct RoiParams
{
	int x1;
	int y1;
	int x2;
	int y2;
} roi_params;


int main(int argc, const char* argv[])
{
	median_params.window_size = 10;
	median_params.partition = 128; // default value
	
	sobel_params.srcType = CV_32F;//CV_8UC1; //CV_64FC1; //CV_8UC1;
	sobel_params.dstType = CV_32F; //CV_8UC1; //CV_64FC1; //CV_8UC1;
	sobel_params.dx = 1;
	sobel_params.dy = 1;
	sobel_params.ksize = 5;
	sobel_params.scale = 1;		// Default value, no scaling.
	
	canny_params.low_thresh = 2.0;
	canny_params.high_thresh = 200.0;
	canny_params.apperture_size = 5;
	
	roi_params.x1 = 375;
	roi_params.y1 = 150;
	roi_params.x2 = 850;
	roi_params.y2 = 675;
	
	cv::Rect roi(roi_params.x1, roi_params.y1,
		roi_params.x2 - roi_params.x1,
		roi_params.y2 - roi_params.y1);
	
	lsnake_params.border_clue = std::vector<cv::Point>();
	lsnake_params.border_clue.push_back(cv::Point(495 - roi_params.x1, 240 - roi_params.y1));
	lsnake_params.border_clue.push_back(cv::Point(731 - roi_params.x2, 240 - roi_params.y1));
	lsnake_params.border_clue.push_back(cv::Point(731 - roi_params.x2, 630 - roi_params.y2));
	lsnake_params.border_clue.push_back(cv::Point(495 - roi_params.x1, 630 - roi_params.y2));
	lsnake_params.max_distance = 14;
	lsnake_params.min_distance = 6;
	lsnake_params.min_angle = 33;
	lsnake_params.rings = 21;
	
	
	bool use_sobel = false;

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
	cv::namedWindow("Blur", cv::WINDOW_OPENGL);  cv::moveWindow("Blur", 480, 50);
	cv::namedWindow("Edges", cv::WINDOW_OPENGL);  cv::moveWindow("Edges", 480, 400);
	cv::namedWindow("Mod", cv::WINDOW_OPENGL);  cv::moveWindow("Mod", 10, 400);
	cv::cuda::setGlDevice();

	// Ask if device is compatible
	cv::cuda::DeviceInfo dev_info = cv::cuda::DeviceInfo();
	bool bglcomp = dev_info.isCompatible();
	std::cout << "This computer's device compatibility with GlDevice is " << bglcomp << std::endl;
	if (!bglcomp) return -1;

	cv::cuda::GpuMat d_frame, d_roi, d_frame_view;
#ifdef DEBUG_SHOW
	cv::cuda::GpuMat d_frame_edges;
#endif
	cv::Ptr<cv::cudacodec::VideoReader> d_reader = cv::cudacodec::createVideoReader(fname);
	cv::cuda::GpuMat d_blur, d_dst;

	cv::cudacodec::FormatInfo formatInfo = d_reader->format();
	std::cout << formatInfo.width << "x" << formatInfo.height << " codec:" << formatInfo.codec << std::endl;
	
	cv::Ptr<cv::cuda::Filter> median_blur = cv::cuda::createMedianFilter (CV_8UC1,
											median_params.window_size,
											median_params.partition);

	cv::Ptr<cv::cuda::Filter> sobel_edg_x, sobel_edg_y;
	cv::cuda::GpuMat d_float, d_sobel_x, d_sobel_y;
	
	cv::Ptr<cv::cuda::CannyEdgeDetector> canny_edg;
	if (use_sobel)
	{
		std:: cout << "+++ Using sobel" << std::endl;
		sobel_edg_x = cv::cuda::createSobelFilter(
				sobel_params.srcType,
				sobel_params.dstType,
				sobel_params.dx,
				sobel_params.dy,
				sobel_params.ksize,
				sobel_params.scale);
		sobel_edg_y = cv::cuda::createSobelFilter(
				sobel_params.srcType,
				sobel_params.dstType,
				sobel_params.dx,
				sobel_params.dy,
				sobel_params.ksize,
				sobel_params.scale);
		std:: cout << "+++ Using sobel created..." << std::endl;
	}
	else
	{
		std:: cout << "+++ Using canny" << std::endl;
		canny_edg = cv::cuda::createCannyEdgeDetector(canny_params.low_thresh,
				canny_params.high_thresh,
				canny_params.apperture_size, false);
	}

	LinearSplineTracker lspline_tracker = LinearSplineTracker(
		lsnake_params.border_clue,
		lsnake_params.max_distance,
		lsnake_params.min_distance,
		lsnake_params.min_angle,
		lsnake_params.rings);
	
	int nframe = 0;
	int ndiff = 0;
	for (;;)
	{
		// Read frame in original video
		if (!d_reader->nextFrame(d_frame))
			break;

		d_roi = cv::cuda::GpuMat(d_frame, roi);
		cv::cuda::cvtColor(d_roi, d_blur, CV_BGRA2GRAY);
		
		median_blur->apply(d_blur, d_blur);
#ifdef DEBUG_SHOW
		cv::imshow("Blur", d_blur);
#endif
		
		// http://docs.opencv.org/3.2.0/d0/d05/group__cudaimgproc.html
		if (use_sobel)
		{
			//A.convertTo(B,CV_8U,255.0/(Max-Min),-255.0*Min/(Max-Min));
			d_blur.convertTo(d_float, CV_32F);
			
			sobel_edg_x->apply(d_float, d_sobel_x);
			sobel_edg_y->apply(d_float, d_sobel_y);
			cv::cuda::abs(d_sobel_x, d_sobel_x);
			cv::cuda::abs(d_sobel_y, d_sobel_y);
			cv::cuda::addWeighted(d_sobel_x, 0.5, d_sobel_y, 0.5, 0, d_float);
			cv::cuda::abs(d_float, d_float);
			d_float.convertTo(d_dst, CV_8UC1);
		}
		else
		{
			canny_edg->detect(d_blur, d_dst);
		}
		//cv::imshow("Mod", d_dst);
#ifdef DEBUG_SHOW
		d_roi.copyTo(d_frame_edges, d_dst);
		cv::imshow("Edges", d_frame_edges);
#else
		cv::imshow("Edges", d_dst);
#endif		
		d_roi.copyTo(d_frame_view);
		lspline_tracker.track(d_roi, d_dst, d_frame_view);
		cv::imshow("GPU", d_frame);
		cv::imshow("Mod", d_frame_view);
#ifdef DEBUG_SHOW
		d_frame_edges.setTo(cv::Scalar(0));
#endif
		
        if (cv::waitKey(-1) == 'q')
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
