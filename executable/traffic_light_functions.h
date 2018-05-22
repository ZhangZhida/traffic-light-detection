//
// Created by zhida on 18-5-22.
//


#ifndef TRAFFIC_LIGHT_DETECTION_TRAFFIC_LIGHT_FUNCTIONS_H
#define TRAFFIC_LIGHT_DETECTION_TRAFFIC_LIGHT_FUNCTIONS_H

#include <opencv2/opencv.hpp>
#include "FileOperator.h"

using namespace std;
using namespace cv;

std::vector<std::string> readFileList(char *basePath);

Mat ContourOperation(Mat canny_output, vector<vector<Point> >& contours, vector<Vec4i> hierarchy, RNG rng);

void CannyOperation(Mat& src, Mat& src_gray, Mat& dst, Mat& detect_edges, int lowThreshold, int ratio, int kernal_size);

void my_imshow(string name, Mat img);

void my_imwrite(string name, Mat img);

Rect rectCenterScale(Rect rect0, double ratio_width, double ratio_height);

void houghLinesAnalyze(vector<Vec2f> lines, vector<Vec2f>& resultLines);

void color_filter(Mat img_src_HSV, Mat& red_yellow_green_hue_range);

// 显示Hue图上找指定颜色的效果
void show_hue_roi_area(Mat img_src_gray, Mat red_yellow_green_hue_range, string filename);

void center_scale(Rect &rect0, Mat img_src);

void ROI_canny_operation(Mat img_src_hue_range, Mat& detected_edges);

void houghlines_operation(vector<Vec2f> lines, double traffic_light_shorter_side, int MIN_HOUGHLINE_THRESHOLD, Mat detected_edges, Rect rect0, vector<Rect>& object_rect_vec, int index);

void crop_areas(vector<Rect> object_rect_vec, Mat img_src);

#endif //TRAFFIC_LIGHT_DETECTION_TRAFFIC_LIGHT_FUNCTIONS_H


