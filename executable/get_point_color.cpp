//
// Created by zhida on 18-5-22.
//
#include <opencv2/opencv.hpp>
#include "../include/MouseGetPoint.h"

using namespace std;
using namespace cv;

int main() {

    Mat img = imread("/home/zhida/Desktop/example_data/2018_0318_153903_894.mp4_20180329_162437.151.jpg");

    getColor(img, 0.5, 1);

}