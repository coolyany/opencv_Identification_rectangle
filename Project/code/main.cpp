#include <iostream>
#include <vector>
#include <math.h>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

#define OverGap			20

inline void cvShow(string name, Mat img)
{
	cv::imshow(name, img);
	//cv::waitKey(0);
}

/* 判断顶点是否重叠 */
inline bool selectOverPoint(std::vector<cv::Point> prePoints, std::vector<cv::Point> lastPoints)
{
	//默认矩形四个顶点
	for (size_t i = 0; i < 4; i++)
	{
		std::cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << std::endl;
		std::cout << "prePoints : " << prePoints[i] << "lastPoints : " << lastPoints[i] << std::endl;
		std::cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << std::endl;

		double dx = fabs(prePoints[i].x - lastPoints[i].x);
		double dy = fabs(prePoints[i].y - lastPoints[i].y);
		//判断顶点之间相差OverGap个像素点以内为重叠
		if (dx < OverGap && dy < OverGap)
		{
			continue;
		}
		else {
			return false;
		}
	}
	return true;
}
/* 判断矩形位置是否重叠 */
inline std::vector<std::vector<cv::Point>> selectOverRect(std::vector<std::vector<cv::Point>> &squares)
{
	std::vector<std::vector<cv::Point>> targetDst;
	targetDst.clear();

	targetDst.push_back(squares[0]);
	for (size_t i = 0; i < squares.size() - 1; i++)
	{
		//若为重叠矩形则抛弃
		if (selectOverPoint(squares[i], squares[i + 1]))
		{
			continue;
		}
		else
		{
			targetDst.push_back(squares[i + 1]);
		}
	}

	return targetDst;
}

double angle(cv::Point pt1, const cv::Point &pt2, const cv::Point &pt0)
{
	double dx1 = pt1.x - pt0.x;
	double dy1 = pt1.y - pt0.y;
	double dx2 = pt2.x - pt0.x;
	double dy2 = pt2.y - pt0.y;
	return (dx1*dx2 + dy1 * dy2) / sqrt((dx1*dx1 + dy1 * dy1)*(dx2*dx2 + dy2 * dy2) + 1e-10);
}

bool isRepeatRect(std::vector<cv::Point> prePoint, std::vector<cv::Point> lastPoint)
{
	double dx = 0, dy = 0;
	//矩形4个点
	for (size_t i = 0; i < 4; i++)
	{
		dx = fabs(prePoint[i].x - lastPoint[i].x);
		dy = fabs(prePoint[i].y - lastPoint[i].y);
		if (dx < 5 && dy < 5)
		{

		}
	}
	return false;
}

void findRect(const cv::Mat &img, cv::Mat &out)
{
	int thresh = 50, N = 5;
	std::vector<std::vector<cv::Point>> squares;
	squares.clear();//初始化容器

	cv::Mat src, dst, gray_one, gray;
	src = img.clone();
	out = img.clone();
	gray_one = cv::Mat(src.size(), CV_8U);
	//imshow("gray", gray_one);

	//滤波增强边缘检测,中值滤波
	medianBlur(src, dst, 9);
	bilateralFilter(src, dst, 25, 25 * 2, 35);
	//GaussianBlur(src, dst, cv::Size(7, 7), 0, 0);


	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	int index = 0, targetNum = 0;

	//在图像的每个颜色通道中查找矩形
	std::cout << "image channels :: " << img.channels() << std::endl;
	for (int c = 0; c < img.channels(); c++)
	{
		int ch[] = { c, 0 };
		//通道分离
		mixChannels(&dst, 1, &gray_one, 1, ch, 1);
		imshow("gray", gray_one);
		// 尝试几个阈值
		for (int l = 0; l < N; l++)
		{
			// 用canny()提取边缘
			if (l == 0)
			{
				//检测边缘
				cv::Canny(gray_one, gray, 5, thresh, 5);
				//膨脹
				//cv::dilate(gray, gray, cv::Mat(), cv::Point(-1, -1));
				imshow("dilate", gray);
			}
			else
			{
				//二值图像
				gray = gray_one >= (l + 1) * 255 / N;
				//imshow("gray", gray);
			}

			// 轮廓查找
			findContours(gray, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
			findContours(gray, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

			std::vector<cv::Point> approx;

			// 检测所找到的轮廓
			for (size_t i = 0; i < contours.size(); i++)
			{
				//使用图像轮廓点进行多边形拟合
				 //contours[i]代表的是第i个轮廓，contours[i].size()代表的是第i个轮廓上所有的像素点数
				cv::approxPolyDP(cv::Mat(contours[i]), approx, cv::arcLength(cv::Mat(contours[i]), true)*0.02, true);


				//std::cout << "i :: " << approx.size() << std::endl;
				/* 
					approx.size() 判断是否有4个拐点
					contourArea 判断矩形面积范围
					isContourConvex 判定一个轮廓是否是凸包,是返回true,否则返回false
				*/
				if (approx.size() == 4 && fabs(cv::contourArea(cv::Mat(approx))) > 100 && cv::isContourConvex(cv::Mat(approx)))
				{

					index++;
					double maxCosine = 0;

					/*std::cout << "approx[0]  " << approx[0] << std::endl;
					std::cout << "approx[1]  " << approx[1] << std::endl;
					std::cout << "approx[2]  " << approx[2] << std::endl;
					std::cout << "approx[3]  " << approx[3] << std::endl;*/

					//遍历三次
					for (int j = 2; j < 5; j++)
					{
						// 求轮廓边缘之间角度的最大余弦,取三个点求角度
						double cosine = fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
						maxCosine = MAX(maxCosine, cosine);
						std::cout << "maxCosine " << maxCosine << std::endl;
					}

					//误差
					if (maxCosine < 0.3)
					{
						squares.push_back(approx);
						targetNum++;
					}
				}
			}
		}
	}
	std::cout << "index :: " << index << std::endl;
	std::cout << "targetNum :: " << targetNum << std::endl;
	std::cout << "squares num :: " << squares.size() << std::endl;

	for (size_t k = 0; k < squares.size(); k++)
	{
		std::cout << "squares :: " << squares[k][0] << std::endl;
		/*std::cout << "squares[1]  " << squares[1] << std::endl;
		std::cout << "squares[2]  " << squares[2] << std::endl;
		std::cout << "squares[3]  " << squares[3] << std::endl;*/
	}

	std::vector<std::vector<cv::Point>> targetRect = selectOverRect(squares);

	for (size_t i = 0; i < targetRect.size(); i++)
	{
		const cv::Point* p = &targetRect[i][0];

		int n = (int)targetRect[i].size();
		if (p->x > 3 && p->y > 3)
		{
			//画出目标框
			polylines(out, &p, &n, 1, true, cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
		}
	}
	std::string stdNumText = "Number of rectangles :: " + std::to_string(targetRect.size());
	cv::putText(out, stdNumText, cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 23, 0), 2, 8);
	cv::namedWindow("out", cv::WINDOW_NORMAL);
	cv::imshow("out", out);
}


void findRect2(const cv::Mat &img, cv::Mat &out)
{
	cv::Mat gray_img, gauss_img, threshold_img, ContrastImg;
	cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);
	//cvShow("gray_img", gray_img);
	cv::GaussianBlur(gray_img, gauss_img, Size(5, 5), 1);
	//cvShow("gauss_img", gauss_img);
	//4, 阈值处理
	cv::threshold(gauss_img, threshold_img, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
	//cvShow("thresholdImg", threshold_img);
	
	//5，找轮廓
	vector<vector<cv::Point>> contours;
	cv::findContours(threshold_img, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

	Mat drawImg = img.clone();
	cv::drawContours(drawImg, contours, -1, Scalar(0, 0, 255), 2);//画轮廓
	cvShow("drawImg", drawImg);

	std::vector<cv::Point> approx;
	int index = 0, targetNum = 0;
	std::vector<std::vector<cv::Point>> squares;

	// 检测所找到的轮廓
	for (size_t i = 0; i < contours.size(); i++)
	{
		//使用图像轮廓点进行多边形拟合
		 //contours[i]代表的是第i个轮廓，contours[i].size()代表的是第i个轮廓上所有的像素点数
		cv::approxPolyDP(cv::Mat(contours[i]), approx, cv::arcLength(cv::Mat(contours[i]), true)*0.02, true);
		/***
		 *** approx.size() 判断是否有4个拐点
		 *** contourArea 判断矩形面积范围
		 *** isContourConvex 判定一个轮廓是否是凸包,是返回true,否则返回false
		***/
		if (approx.size() == 4 && fabs(cv::contourArea(cv::Mat(approx))) > 100 && cv::isContourConvex(cv::Mat(approx)))
		{

			index++;
			double maxCosine = 0;
			//遍历三次
			for (int j = 2; j < 5; j++)
			{
				// 求轮廓边缘之间角度的最大余弦,取三个点求角度
				double cosine = fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
				maxCosine = MAX(maxCosine, cosine);
				std::cout << "maxCosine " << maxCosine << std::endl;
			}

			//角度误差
			if (maxCosine < 0.3)
			{
				squares.push_back(approx);
				targetNum++;
			}
		}
	}
	std::cout << "squares num :: " << squares.size() << std::endl;
	for (size_t k = 0; k < squares.size(); k++)
	{
		std::cout << "squares :: " << squares[k] << std::endl;
		/*std::cout << "squares[1]  " << squares[1] << std::endl;
		std::cout << "squares[2]  " << squares[2] << std::endl;
		std::cout << "squares[3]  " << squares[3] << std::endl;*/
	}

	std::vector<std::vector<cv::Point>> targetRect = selectOverRect(squares);

	std::cout << "index :: " << index << std::endl;
	std::cout << "targetNum :: " << targetNum << std::endl;
	std::cout << "squares num :: " << targetRect.size() << std::endl;
	for (size_t k = 0; k < targetRect.size(); k++)
	{
		std::cout << "squares :: " << targetRect[k] << std::endl;
	}
	
	cv::Mat target = img.clone();
	cv::drawContours(target, targetRect, -1, Scalar(0, 0, 255), 2);//画轮廓
	
	std::string stdNumText = "Number of rectangles :: " + std::to_string(targetRect.size());
	cv::putText(target, stdNumText, cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 2, 8);
	cvShow("out", target);
}

int main(int argc, char* argv[])
{
	//cv::Mat pic = cv::imread("D:\\picture\\6.jpg");
	//std::cout << pic.channels() << std::endl;
	//cv::imshow("test", pic);

	cv::Mat src, dst;
	src = cv::imread("D:\\Code\\opencv\\opencv_rect\\Project\\picture\\1.jpg");
	/*cv::imshow("src", src);
	cv::namedWindow("src", cv::WINDOW_NORMAL);
	findRect(src, dst);
	cv::imshow("out", src);
	cv::namedWindow("out", cv::WINDOW_NORMAL);*/
	cv::imshow("src", src);
	cv::namedWindow("src", cv::WINDOW_NORMAL);
	findRect(src, dst);


	cv::waitKey();
}