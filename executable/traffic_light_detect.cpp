#include <opencv2/opencv.hpp>
#include "../include/traffic_light_functions.h"
#include "../include/MouseGetPoint.h"

using namespace std;
using namespace cv;

#define MIN_HOUGHLINE_THRESHOLD 15


int main(){

    char* inputDir = "/home/zhida/Desktop/example_data";
    vector<string> fileList;


//#define TEST_SINGLE_DIR
    fileList.push_back("2018_0318_153903_894.mp4_20180329_162454.752.jpg");
#ifndef TEST_SINGLE_DIR
    fileList = readFileList(inputDir);
#endif //TEST_WHOLE_DIR


    for (auto filename:fileList) {

        cout << "==================================" << endl << filename << endl;
        string filepath = string(inputDir) + "/" + filename;

        Mat img_src;
        Mat img_src_gray;
        Mat img_src_HSV;
        Mat red_yellow_green_hue_ROI;
        RNG rng(12345);
        vector<Vec4i> hierarchy_hue;

        // 读入图片
        img_src = cv::imread(filepath);
        // convert BGR -> GRAY
        cvtColor(img_src, img_src_gray, CV_BGR2GRAY);
        // convert BGR -> HSV
        cvtColor(img_src, img_src_HSV, CV_BGR2HSV);


        //////////
        // 1. 先找出（1）红黄绿值满足一定范围（2）最短边大小满足一定要求的局部区域，得到满足颜色要求的ROI区域
        //////////
        color_filter(img_src_HSV, red_yellow_green_hue_ROI);

        // 在HSV空间中显示红绿黄颜色过滤效果
        show_hue_roi_area(img_src_gray, red_yellow_green_hue_ROI, filename);

        //////////
        // 2. findContours函数在hue空间里的颜色ROI整图上检测轮廓（连通区）
        //////////
        vector<vector<Point> > contours_hue;
        findContours(red_yellow_green_hue_ROI, contours_hue, hierarchy_hue, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0,0) );

        // 包围盒
        vector<Rect> object_rect_vec;

        for (int i=0; i<contours_hue.size(); i++) {

            Rect rect0 = boundingRect(Mat(contours_hue[i]));
            double traffic_light_width = rect0.width;
            double traffic_light_shorter_side = traffic_light_width;

            //////////
            // 3.1. 适当扩大包围盒的范围（为了把整个红绿灯都包括进来）
            //////////
            center_scale(rect0, img_src);
            Mat img_src_hue_range = img_src(rect0);

            //////////
            // 3.2. 用Canny算子边缘检测
            //////////
            Mat ROI_detected_edges;
            ROI_canny_operation(img_src_hue_range, ROI_detected_edges);

            //////////
            // 3.3. 用HoughLines函数检测Canny检测结果里的直线
            //////////
            vector<Vec2f> lines;
            int houghLinesThreshlold = (int)(traffic_light_shorter_side * 1.5) > MIN_HOUGHLINE_THRESHOLD ? (int)(traffic_light_shorter_side * 1.5) : MIN_HOUGHLINE_THRESHOLD;
            HoughLines(ROI_detected_edges, lines, 1.2, 1.0 * CV_PI / 180, houghLinesThreshlold, 0, 0);

            //////////
            // 3.4. 对诸多竖直的直线进行筛选，这里我假设直线主要聚集在红绿灯外框的边缘，
            //      所以用kmeans聚类算法，选出两条竖直直线，作为红绿灯外框的左、右边界；对水平直线进行同样操作，找出红绿灯外框的上、下边界,
            //      并把结果存入object_rect_vec
            //////////
            houghlines_operation(lines, traffic_light_shorter_side, MIN_HOUGHLINE_THRESHOLD, ROI_detected_edges, rect0, object_rect_vec, i);

        }

        for (int i=0; i<object_rect_vec.size(); i++) {

            rectangle(img_src, object_rect_vec[i].tl(), object_rect_vec[i].br(), Scalar(0,0,255), 2);
            my_imshow("img_src", img_src);
        }

        //////////
        // 3.5. crop上面的上、下、左、右边界中的区域————疑似红绿灯区域，作为下一步的分类网的输入
        //////////
        crop_areas(object_rect_vec, img_src);



        cv::waitKey(0);

    }


    return 0;
}



