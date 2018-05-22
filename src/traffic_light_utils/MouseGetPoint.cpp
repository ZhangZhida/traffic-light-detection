#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include "../../include/MouseGetPoint.h"

using namespace cv;
using namespace std;

Point current_pt;
void mouseHandlerColor(int event, int x, int y, int flags, void *param) {
    switch (event) {
        case CV_EVENT_LBUTTONDOWN:
            current_pt = Point(x, y);
    }
}

void getColor(const cv::Mat &img, double scale, int mode) {

    //namedWindow("img", CV_WINDOW_AUTOSIZE);
    Mat _img = img.clone();
    Mat showImg;
    cv::resize(_img, showImg, cv::Size(int(_img.cols / scale), int(_img.rows / scale)));

    if(mode == 0){
        //rgb mode
        imshow("image",showImg);
        setMouseCallback("image", mouseHandlerColor, NULL);
        waitKey(0);
        int b,g,r;
        b=showImg.at<Vec3b>(current_pt.y,current_pt.x)[0];
        g=showImg.at<Vec3b>(current_pt.y,current_pt.x)[1];
        r=showImg.at<Vec3b>(current_pt.y,current_pt.x)[2];
        cout<<" b g r = Scalar("<<b<<","<<g<<","<<r<<")"<<endl;
        cout<<" x y = "<<current_pt.x<<" "<<current_pt.y<<endl;
    }else{
        //hsv mode
        Mat hsv;
        cvtColor(showImg, hsv, CV_RGB2HSV);
        imshow("image",showImg);
        setMouseCallback("image", mouseHandlerColor, NULL);
        waitKey(0);
        int h,s,v;
        h=hsv.at<Vec3b>(current_pt.y,current_pt.x)[0];
        s=hsv.at<Vec3b>(current_pt.y,current_pt.x)[1];
        v=hsv.at<Vec3b>(current_pt.y,current_pt.x)[2];
        cout<<" h s v = Scalar("<<h<<","<<s<<","<<v<<")"<<endl;
        cout<<" x y = "<<current_pt.x<<" "<<current_pt.y<<endl;
    }

    cv::destroyWindow("image");
    return;
}