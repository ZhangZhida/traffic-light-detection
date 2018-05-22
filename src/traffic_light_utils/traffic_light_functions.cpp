//
// Created by zhida on 18-5-22.
//

#include "traffic_light_functions.h"



std::vector<std::string> readFileList(char *basePath)
{
    std::vector<std::string> result;
    DIR *dir;
    struct dirent *ptr;
    char base[1000];

    if ((dir=opendir(basePath)) == NULL)
    {
        perror("Open dir error...");
        exit(1);
    }

    while ((ptr=readdir(dir)) != NULL)
    {
        if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0)
            continue;
        else if(ptr->d_type == 8)    ///file
        {printf("d_name:%s/%s\n",basePath,ptr->d_name);
            result.push_back(std::string(ptr->d_name));}
        else if(ptr->d_type == 10)    ///link file
        {printf("d_name:%s/%s\n",basePath,ptr->d_name);
            result.push_back(std::string(ptr->d_name));}
        else if(ptr->d_type == 4)    ///dir
        {
            memset(base,'\0',sizeof(base));
            strcpy(base,basePath);
            strcat(base,"/");
            strcat(base,ptr->d_name);
            result.push_back(std::string(ptr->d_name));
            readFileList(base);
        }
    }
    closedir(dir);
    return result;
}

Mat ContourOperation(Mat canny_output, vector<vector<Point> >& contours, vector<Vec4i> hierarchy, RNG rng) {

    // Find coutours
//    findContours(canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
    findContours(canny_output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0,0));

    // Draw contours
    Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
    for(int i=0; i<contours.size(); i++) {
        Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
    }

    return drawing;
}


void CannyOperation(Mat& src, Mat& src_gray, Mat& dst, Mat& detect_edges, int lowThreshold, int ratio, int kernal_size){

    blur(src_gray, detect_edges, Size(2.5,2.5));

    Canny(detect_edges, detect_edges, lowThreshold, lowThreshold*ratio, kernal_size);

    //dst = Scalar::all(0);

    src.copyTo(dst, detect_edges);

}

void my_imshow(string name, Mat img){

    namedWindow(name, WINDOW_AUTOSIZE);
    imshow(name, img);
}

void my_imwrite(string name, Mat img) {

    cv::imwrite(name, img);
//    cout << "image wrote to" + name << endl;
}

//void MatchingMethod(int, void*){
//}

Rect rectCenterScale(Rect rect0, double ratio_width, double ratio_height) {

    Point center = (rect0.tl() + rect0.br())/ 2;

    double width = rect0.size().width;
    double height =  rect0.size().height;

    // scale
    rect0 = rect0 + Size(width * ratio_width, height * ratio_height);

    // move
    rect0 = rect0 + Point( - width * ratio_width / 2, - height * ratio_height / 2);

    return rect0;
}


void houghLinesAnalyze(vector<Vec2f> lines, vector<Vec2f>& resultLines) {

    vector<float> thetaColl_vertical, rhoColl_vertical;
    vector<float> thetaColl_horizontal, rhoColl_horizontal;
    for (int j=0; j<lines.size();j++){

        float rho = lines[j][0], theta = lines[j][1];

        // 要求theta角大约在PI/2左右 —> 竖直
        if(theta > 1.55 && theta < 1.59) {

            thetaColl_vertical.push_back(theta);
            rhoColl_vertical.push_back(rho);
        }

        // 要求theta角大约在0或者PI左右 —> 水平情况
        if(theta < 0.02 || theta > 3.13) {

//            cout << "theta: " << theta << ", rho: " << rho << endl;
            thetaColl_horizontal.push_back(theta);
            rhoColl_horizontal.push_back(rho);
        }
    }

    // 判断是否同时有水平直线和竖直直线
    if(thetaColl_horizontal.size() < 2 || thetaColl_vertical.size() < 2) {
        return;
    }


    Mat rhoColl_hori_mat = Mat(rhoColl_horizontal);
    Mat rhoColl_vert_mat = Mat(rhoColl_vertical);
    Mat labels_mat;
    Mat centers_horizontal;
    Mat centers_vertical;

    // kmeans聚类算法，对多条直线的rho值进行聚类，得到多个rho值的两个中心，存在centers_horizontal和centers_vertical里
    kmeans(rhoColl_hori_mat, 2, labels_mat, TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0), 100, KMEANS_PP_CENTERS, centers_horizontal);
    kmeans(rhoColl_vert_mat, 2, labels_mat, TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0), 100, KMEANS_PP_CENTERS, centers_vertical);


//    cout << "labels_mat: " << labels_mat << endl;
//    cout << "centers_horizontal: " << centers_horizontal << endl;
//    cout << "centers_vertical" << centers_vertical << endl;

    Vec2f resultLine_hori_1 = Vec2f((centers_horizontal.at<float>(0,0)), 0.01);
    Vec2f resultLine_hori_2 = Vec2f((centers_horizontal.at<float>(1,0)), 0.01);
    Vec2f resultLine_vert_1 = Vec2f((centers_vertical.at<float>(0,0)), 1.57);
    Vec2f resultLine_vert_2 = Vec2f((centers_vertical.at<float>(1,0)), 1.57);

    resultLines.push_back(resultLine_hori_1);
    resultLines.push_back(resultLine_hori_2);
    resultLines.push_back(resultLine_vert_1);
    resultLines.push_back(resultLine_vert_2);

//    cout << "resultLine_hori_1" << resultLine_hori_1 << endl;
//    cout << "resultLine_hori_2" << resultLine_hori_2 << endl;
}

void color_filter(Mat img_src_HSV, Mat& red_yellow_green_hue_range){

    Mat lower_red_hue_range;
    Mat upper_red_hue_range;
    Mat green_hue_range;
    Mat yellow_hue_range;
    Mat black_hue_range;


    // 加上红色出现区域
    cv::inRange(img_src_HSV, cv::Scalar(0, 100, 100), cv::Scalar(10, 256, 256), lower_red_hue_range);
    cv::inRange(img_src_HSV, cv::Scalar(160, 100, 100), cv::Scalar(179, 256, 256), upper_red_hue_range);
    cv::addWeighted(lower_red_hue_range, 1.0, upper_red_hue_range, 1.0, 0.0, red_yellow_green_hue_range);

    // 加上绿色出现区域
    cv::inRange(img_src_HSV, cv::Scalar(32, 50, 100), cv::Scalar(77, 256, 256), green_hue_range);
    cv::addWeighted(green_hue_range, 1.0, red_yellow_green_hue_range, 1.0, 0.0, red_yellow_green_hue_range);

    // 加上黄色出现区域
    cv::inRange(img_src_HSV, cv::Scalar(22, 100, 100), cv::Scalar(38, 256, 256), yellow_hue_range);
    cv::addWeighted(yellow_hue_range, 1.0, red_yellow_green_hue_range, 1.0, 0.0, red_yellow_green_hue_range);

//    // 加上黑色出现区域
//    cv::inRange(img_src_HSV, cv::Scalar(0, 0, 0), cv::Scalar(179, 255, 46), black_hue_range);
//    cv::addWeighted(black_hue_range, 1.0, red_yellow_green_hue_range, 1.0, 0.0, red_yellow_green_hue_range);
//
//    my_imshow("red_yellow_green_hue_range", red_yellow_green_hue_range);
}

void show_hue_roi_area(Mat img_src_gray, Mat red_yellow_green_hue_range, string filename){

    Mat effect_hue_bgr;
    img_src_gray.copyTo(effect_hue_bgr);
    cvtColor(img_src_gray, effect_hue_bgr, CV_GRAY2BGR);


    Mat red_yellow_green_hue_range_bgr;
    cvtColor(red_yellow_green_hue_range, red_yellow_green_hue_range_bgr, CV_GRAY2BGR);

    int nrows = red_yellow_green_hue_range_bgr.rows;
    int ncols = red_yellow_green_hue_range_bgr.cols;
    for (int i=0;i<nrows;i++)
    {
        for (int j=0;j<ncols;j++)
        {
            red_yellow_green_hue_range_bgr.at<cv::Vec3b>(i,j)[0]=red_yellow_green_hue_range_bgr.at<cv::Vec3b>(i,j)[0] * 100;  //B
            red_yellow_green_hue_range_bgr.at<cv::Vec3b>(i,j)[1]=0;    //G
            red_yellow_green_hue_range_bgr.at<cv::Vec3b>(i,j)[2]=0;    //R
        }
    }

    cv::addWeighted(effect_hue_bgr, 1.0, red_yellow_green_hue_range_bgr, 1.0, 0.0, effect_hue_bgr);

    my_imshow("effect_hue_original", effect_hue_bgr);
    my_imwrite("/home/zhida/Documents/Code/traffic-light-detection/resource/result/"+filename + "_hue.jpg", effect_hue_bgr);

}

void center_scale(Rect &rect0, Mat img_src){

    // center scale
    rect0 = rectCenterScale(rect0, 1.5, 6);
    if(rect0.tl().x < 0)
        rect0.x = 0;
    if (rect0.tl().y <0)
        rect0.y = 0;
    if (rect0.br().x > img_src.size().width)
        rect0.width = img_src.size().width - rect0.tl().x;
    if(rect0.br().y > img_src.size().height)
        rect0.height = img_src.size().height - rect0.tl().y ;
    rect0 = Rect(rect0.x, rect0.y, rect0.width, rect0.height);
}

void ROI_canny_operation(Mat img_src_hue_range, Mat& detected_edges){

    Mat img_src_hue_range_gray;
    cvtColor(img_src_hue_range, img_src_hue_range_gray, CV_BGR2GRAY);

    int lowThreshold = 55;
    int ratio = 3;
    int canny_kernal_size = 3;
    Mat canny_detected_edges;

    // Canny
    Mat img_dst_hue_range;
    img_dst_hue_range.create(img_src_hue_range.size(), img_dst_hue_range.type());
    CannyOperation(img_src_hue_range, img_src_hue_range_gray, img_dst_hue_range, detected_edges, lowThreshold, ratio, canny_kernal_size);
}

void houghlines_operation(vector<Vec2f> lines, double traffic_light_shorter_side, int MIN_HOUGHLINE_THRESHOLD, Mat detected_edges, Rect rect0, vector<Rect>& object_rect_vec, int index) {

    if (!lines.empty()) {

        Mat cdst;
        cvtColor(detected_edges, cdst, CV_GRAY2BGR);

        // 使用kmeans聚类算法对诸多竖直和水平的直线进行筛选，这里假设：对于竖直或者水平情况，直线分为了两簇，用kmeans得到两簇直线的中心直线作为结果
        vector<Vec2f> resultLines;
        houghLinesAnalyze(lines, resultLines);
        lines = resultLines;

        // 检查有两条竖直直线和两条水平直线
        if (!lines.empty() && lines.size() == 4) {

            vector<double> object_coord_vec_x_relative;
            vector<double> object_coord_vec_y_relative;
            for( size_t i = 0; i < lines.size(); i++ ){

                float rho = lines[i][0], theta = lines[i][1];
                Point pt1, pt2;
                double a = cos(theta), b = sin(theta);
                double x0 = a*rho, y0 = b*rho;
                vector<double> object_coord;

                object_coord_vec_x_relative.push_back(x0);
                object_coord_vec_y_relative.push_back(y0);

                pt1.x = cvRound(x0 + 1000*(-b));
                pt1.y = cvRound(y0 + 1000*(a));
                pt2.x = cvRound(x0 - 1000*(-b));
                pt2.y = cvRound(y0 - 1000*(a));
                line( cdst, pt1, pt2, Scalar(255, 255, 0),1, CV_AA);
            }

            int base_tl_x = rect0.tl().x;
            int base_tl_y = rect0.tl().y;

            std::sort(object_coord_vec_x_relative.begin(), object_coord_vec_x_relative.end());
            std::sort(object_coord_vec_y_relative.begin(), object_coord_vec_y_relative.end());

            Point object_rect_tl = Point(object_coord_vec_x_relative[2] + base_tl_x, object_coord_vec_y_relative[2] + base_tl_y);
//                Point object_rect_br = Point(object_coord_vec_x_relative.back(), object_coord_vec_y_relative.back());

            int object_rect_width = (int)object_coord_vec_x_relative[3] - object_coord_vec_x_relative[2];
            int object_rect_height = (int)object_coord_vec_y_relative[3] - object_coord_vec_y_relative[2];

            Rect object_rect(object_rect_tl.x,object_rect_tl.y, object_rect_width, object_rect_height);
            object_rect_vec.push_back(object_rect);

            my_imshow("cdst" + to_string(index), cdst);
        }

    }
}

void crop_areas(vector<Rect> object_rect_vec, Mat img_src) {

    for(int i=0; i<object_rect_vec.size(); i++){

        Rect object_rec = object_rect_vec[i];
        Mat object_mat = img_src(object_rec);

        if (!object_mat.empty()) {
            string outfilename = to_string(i) + ".jpg";
            imwrite("/home/zhida/Documents/Code/traffic-light-detection/resource/result/"+outfilename, object_mat);
        }
    }
}
