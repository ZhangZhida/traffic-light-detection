//
// Created by zhida on 18-5-22.
//


#ifndef TRAFFIC_LIGHT_DETECTION_TRAFFIC_LIGHT_FUNCTIONS_H
#define TRAFFIC_LIGHT_DETECTION_TRAFFIC_LIGHT_FUNCTIONS_H

#include <opencv2/opencv.hpp>
#include "FileOperator.h"

using namespace std;
using namespace cv;

// 读文件夹，返回文件列表（只有文件名，没有路径信息）
std::vector<std::string> readFileList(char *basePath);

// 找到Contours，并绘制出来
Mat ContourOperation(Mat canny_output, vector<vector<Point> >& contours, vector<Vec4i> hierarchy, RNG rng);

// 先高斯模糊，再Canny检测，结果存放在detect_edges里面
void CannyOperation(Mat& src, Mat& src_gray, Mat& dst, Mat& detect_edges, int lowThreshold, int ratio, int kernal_size);

// 中心缩放
Rect rectCenterScale(Rect rect0, double ratio_width, double ratio_height);

// 对霍夫直线检测的结果用kmeans处理成两条竖直直线和两条水平直线
void houghLinesAnalyze(vector<Vec2f> lines, vector<Vec2f>& resultLines);

// 按照红绿黄，在HSV空间内进行过滤，结果存在red_yellow_green_hue_range里面
void color_filter(Mat img_src_HSV, Mat& red_yellow_green_hue_range);

// 在Hue图上显示按指定颜色（红绿黄）过滤的效果
void show_hue_roi_area(Mat img_src_gray, Mat red_yellow_green_hue_range, string filename);

// 中心放大Rect的范围
void center_scale(Rect &rect0, Mat img_src);

// 对ROI进行Canny检测，检测边缘结果存在detected_edges里面
void ROI_canny_operation(Mat img_src_hue_range, Mat& detected_edges);

// 霍夫直线检测后用kmeans得到水平、竖直各两条直线，结果存在object_rect_vec里面
void houghlines_operation(vector<Vec2f> lines, double traffic_light_shorter_side, int MIN_HOUGHLINE_THRESHOLD, Mat detected_edges, Rect rect0, vector<Rect>& object_rect_vec, int index);

// 切割图片，存到result文件夹里
void crop_areas(vector<Rect> object_rect_vec, Mat img_src);

// imshow
void my_imshow(string name, Mat img);

// imwrite
void my_imwrite(string name, Mat img);

#endif //TRAFFIC_LIGHT_DETECTION_TRAFFIC_LIGHT_FUNCTIONS_H


