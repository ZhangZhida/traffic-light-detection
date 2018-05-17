#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace std;
using namespace cv;


//double ROI_WIDTH;
//double ROI_HEIGHT;
//Rect ROI_RECT;
#define MIN_HOUGHLINE_THRESHOLD 15

void MatchingMethod(int, void*);
//void CannyOperation(Mat& src, Mat& src_gray, Mat& dst, Mat& detect_edges, int lowThreshold, int ratio, int kernal_size);
//void my_imshow(string name, Mat img);
//void my_imwrite(string name, Mat img);
//Mat ContourOperation(Mat canny_output, vector<vector<Point> > contours, vector<Vec4i> hierarchy, RNG rng);
//Rect rectCenterScale(Rect rect0, double ratio_width, double ratio_height);

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
    cout << "image wrote to" + name << endl;
}

void MatchingMethod(int, void*){
}

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
        if(theta > 1.55 && theta < 1.59) {

//            cout << "theta: " << theta << ", rho: " << rho << endl;
            thetaColl_vertical.push_back(theta);
            rhoColl_vertical.push_back(rho);
        }

        if(theta < 0.1 || theta > 3.14) {

//            cout << "theta: " << theta << ", rho: " << rho << endl;
            thetaColl_horizontal.push_back(theta);
            rhoColl_horizontal.push_back(rho);
        }
    }

    // 判断是否同时有水平直线和竖直直线
    if(thetaColl_horizontal.size() < 2 || thetaColl_vertical.size() < 2) {
        return;
    }

    // 对霍夫变换得到的水平线进行预处理，位于上下两侧的线应该具有更大的权比较合理
    Mat thetaColl_horizontal_sorted;
    sort(Mat(thetaColl_horizontal), thetaColl_horizontal_sorted, SORT_ASCENDING);

    // 用Kmeans分别对水平和数值的rho值进行聚类，结果中应该竖直和水平各有两个kmeans中心
//    vector<string> labels;
//    labels.push_back("horizontal");
//    labels.push_back("vertical");


    Mat rhoColl_hori_mat = Mat(rhoColl_horizontal);
    Mat rhoColl_vert_mat = Mat(rhoColl_vertical);
    Mat labels_mat;
    Mat centers_horizontal;
    Mat centers_vertical;

//    if (thetaColl_horizontal.size() == 1)
//        centers_horizontal = Mat(thetaColl_horizontal);
//    else
//        kmeans(rhoColl_hori_mat, 2, labels_mat, TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0), 100, KMEANS_PP_CENTERS, centers_horizontal);
//
//    if (thetaColl_vertical.size() == 1)
//        centers_vertical = Mat(thetaColl_vertical);
//    else
//        kmeans(rhoColl_vert_mat, 2, labels_mat, TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0), 100, KMEANS_PP_CENTERS, centers_vertical);
    kmeans(rhoColl_hori_mat, 2, labels_mat, TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0), 100, KMEANS_PP_CENTERS, centers_horizontal);
    kmeans(rhoColl_vert_mat, 2, labels_mat, TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0), 100, KMEANS_PP_CENTERS, centers_vertical);




//        cout << "labels_mat: " << labels_mat << endl;
    cout << "centers_horizontal: " << centers_horizontal << endl;
    cout << "centers_vertical" << centers_vertical << endl;

    Vec2f resultLine_hori_1 = Vec2f((centers_horizontal.at<float>(0,0)), 0.01);
    Vec2f resultLine_hori_2 = Vec2f((centers_horizontal.at<float>(1,0)), 0.01);
    Vec2f resultLine_vert_1 = Vec2f((centers_vertical.at<float>(0,0)), 1.57);
    Vec2f resultLine_vert_2 = Vec2f((centers_vertical.at<float>(1,0)), 1.57);

    resultLines.push_back(resultLine_hori_1);
    resultLines.push_back(resultLine_hori_2);
    resultLines.push_back(resultLine_vert_1);
    resultLines.push_back(resultLine_vert_2);

//
//    cout << "resultLine_hori_1" << resultLine_hori_1 << endl;
//    cout << "resultLine_hori_2" << resultLine_hori_2 << endl;
}

int main(){

//    Mat img_src = cv::imread("/home/zhida/Desktop/example_data/2018_0318_155903_898.mp4_20180329_174351.047.jpg");
//    Mat img_src = cv::imread("/home/zhida/Desktop/example_data/2018_0318_155903_898.mp4_20180329_174349.380.jpg");
    Mat img_src = cv::imread("/home/zhida/Desktop/example_data/2018_0318_155903_898.mp4_20180329_174350.214.jpg");
//    Mat img_src = cv::imread("/home/zhida/Desktop/example_data/2018_0318_155903_898.mp4_20180329_174342.713.jpg");
//    Mat img_src = cv::imread("/home/zhida/Desktop/example_data/00129.png");
//    Mat img_src = cv::imread("/home/zhida/Desktop/example_data/00165.png");
//    Mat img_src = cv::imread("/home/zhida/Desktop/example_data/00131.png");
//    Mat img_src = cv::imread("/home/zhida/Desktop/example_data/2018_0318_153903_894.mp4_20180329_162454.752.jpg");

//    Mat img_src = cv::imread("/home/zhida/Desktop/example_data/Columbia_Aerial_Jen_campus_city_3000px.jpg");
    Mat img_src_gray;
    Mat img_dst;
    Mat img_src_HSV;
    RNG rng(12345);
    img_dst.create(img_src.size(), img_src.type());

    my_imshow("img_original", img_src);

    cvtColor(img_src, img_src_gray, CV_BGR2GRAY);

    //my_imshow("src_gray", img_src_gray);

    // convert BGR -> HSV
    cvtColor(img_src, img_src_HSV, CV_BGR2HSV);
    //my_imshow("img_src_HSV", img_src_HSV);

    Mat lower_red_hue_range;
    Mat upper_red_hue_range;
    Mat green_hue_range;
    Mat yellow_hue_range;
    Mat red_yellow_green_hue_range;


    // 加上红色出现区域
    cv::inRange(img_src_HSV, cv::Scalar(0, 100, 100), cv::Scalar(10, 255, 255), lower_red_hue_range);
    cv::inRange(img_src_HSV, cv::Scalar(160, 100, 100), cv::Scalar(179, 255, 255), upper_red_hue_range);
    cv::addWeighted(lower_red_hue_range, 1.0, upper_red_hue_range, 1.0, 0.0, red_yellow_green_hue_range);

    // 加上绿色出现区域
    cv::inRange(img_src_HSV, cv::Scalar(35, 100, 100), cv::Scalar(77, 255, 255), green_hue_range);
    cv::addWeighted(green_hue_range, 1.0, red_yellow_green_hue_range, 1.0, 0.0, red_yellow_green_hue_range);

    // 加上黄色出现区域
    cv::inRange(img_src_HSV, cv::Scalar(26, 100, 100), cv::Scalar(34, 255, 255), yellow_hue_range);
    cv::addWeighted(yellow_hue_range, 1.0, red_yellow_green_hue_range, 1.0, 0.0, red_yellow_green_hue_range);


//    my_imshow("red_hue_range", red_hue_range);

    vector<vector<Point> > contours_hue;
    vector<Vec4i> hierarchy_hue;
    //Mat contours_hue_mat = ContourOperation(red_hue_range, contours_hue, hierarchy_hue, rng);
    Mat contours_hue_;
    findContours(red_yellow_green_hue_range, contours_hue, hierarchy_hue, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0,0) );

//    my_imshow("contours_hue_mat", contours_hue_mat);

    // 包围盒

//    vector<Mat> img_src_hue_range_vector;
//    Mat img_src_hue_range_merged;
    for (int i=0; i<contours_hue.size(); i++) {

        Rect rect0 = boundingRect(Mat(contours_hue[i]));
        double traffic_light_width = rect0.width;
        double traffic_light_height = rect0.height;
        double traffic_light_shorter_side = (traffic_light_width < traffic_light_height) ? traffic_light_width : traffic_light_height;

//        cout << "rect0: " << double(rect0.height) / double(rect0.width) << endl;

//        if(double(rect0.height) / double(rect0.width) > 1.2 || double(rect0.height) / double(rect0.width) < 0.8) {
//            continue;
//        }

        // center scale
        rect0 = rectCenterScale(rect0, 1.5, 6);

        Mat img_src_hue_range = img_src(rect0);
//        my_imshow("img_src_hue_range", img_src_hue_range);

        Mat img_src_hue_range_gray;
        cvtColor(img_src_hue_range, img_src_hue_range_gray, CV_BGR2GRAY);
//        img_src_gray = img_src_hue_range_gray;
        //my_imshow("img_src_hue_range_gray", img_src_hue_range_gray);


        int lowThreshold = 55;
        int ratio = 3;
        int canny_kernal_size = 3;
        Mat canny_detected_edges;

        // Canny

        Mat img_dst_hue_range;
        img_dst_hue_range.create(img_src_hue_range.size(), img_dst_hue_range.type());
        CannyOperation(img_src_hue_range, img_src_hue_range_gray, img_dst_hue_range, canny_detected_edges, lowThreshold, ratio, canny_kernal_size);

//        my_imshow("detect_edges", canny_detected_edges);
        //my_imshow("dst", img_dst);

        // Contour Finding and Drawing

        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        Mat drawing_contour = ContourOperation(canny_detected_edges, contours, hierarchy, rng);

//        if (contours.empty()){
//            continue;
//        }

//    imwrite("/home/zhida/Desktop/example_data/contour_drawing.jpg", drawing_contour);

        Mat img_src_with_contour;
        img_src_hue_range.copyTo(img_src_with_contour, drawing_contour);

        //my_imshow("drawing", drawing_contour);

        // Probabilistic Hough Line Transform

        vector<Vec2f> lines;
//        HoughLinesP(canny_detected_edges, lines, 1, CV_PI/180, 20, 30, 10);


        int houghLinesThreshlold = (int)(traffic_light_shorter_side * 1.5) > MIN_HOUGHLINE_THRESHOLD ? (int)(traffic_light_shorter_side * 1.5) : MIN_HOUGHLINE_THRESHOLD;
//        cout << "houghLinesThreshlold: " << houghLinesThreshlold << endl;
        HoughLines(canny_detected_edges, lines, 1.2, 1 * CV_PI / 180, houghLinesThreshlold, 0, 0);

        if (!lines.empty()) {

            Mat cdst;
            cvtColor(canny_detected_edges, cdst, CV_GRAY2BGR);

//            Vec4i l = lines[i];
//            line( cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255,255,0), 2);
//            canny_detected_edges.copyTo(cdst);
//            cdst = Mat(canny_detected_edges.size(), canny_detected_edges.type(), Scalar(255,255,255));
//            for( size_t i = 0; i < lines.size(); i++ )
//            {
//                Vec2f l = lines[i];
//                line( cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
//            }

            vector<Vec2f> resultLines;
            houghLinesAnalyze(lines, resultLines);
            lines = resultLines;

            if (!lines.empty() && lines.size() == 4) {

                for( size_t i = 0; i < lines.size(); i++ ){
                    float rho = lines[i][0], theta = lines[i][1];
                    Point pt1, pt2;
                    double a = cos(theta), b = sin(theta);
                    double x0 = a*rho, y0 = b*rho;
                    pt1.x = cvRound(x0 + 1000*(-b));
                    pt1.y = cvRound(y0 + 1000*(a));
                    pt2.x = cvRound(x0 - 1000*(-b));
                    pt2.y = cvRound(y0 - 1000*(a));
                    line( cdst, pt1, pt2, Scalar(255, 255, 0),1, CV_AA);
                }

                my_imshow("cdst" + to_string(i), cdst);
            }

//            vector<float> thetaColl;
//            vector<float> rhoColl;
//            for (int j=0; j<lines.size();j++){
//
//                float rho = lines[j][0], theta = lines[j][1];
//                if( (theta > 1.55 && theta < 1.59) || (theta < 0.1 || theta > 3.14)) {
//
//                    cout << "theta: " << theta << ", rho: " << rho << endl;
//                    thetaColl.push_back(theta);
//                    rhoColl.push_back(rho);
//                }
//
//
//            }
//
//
//            for(int j=0; j<rhoColl.size(); j++) {
//
//                float rho = rhoColl[j], theta = thetaColl[j];
//                Point pt1, pt2;
//
//                double a = cos(theta), b = sin(theta);
//                double x0 = a*rho, y0 = b*rho;
//                pt1.x = cvRound(x0 + 1000 * (-b));//cvRound，函数的一种，对一个double型的数进行四舍五入
//                pt1.y = cvRound(y0 + 1000 * (a));
//                pt2.x = cvRound(x0 - 1000 * (-b));
//                pt2.y = cvRound(y0 - 1000 * (a));
//                line(cdst, pt1, pt2, Scalar(255, 255, 0), 1, CV_AA);
//            }

        }




//    my_imwrite("/home/zhida/Documents/Code/traffic-light-detection/resource/result/drawing.jpg", drawing_contour);
////
//        my_imshow("img_src_with_contour", img_src_with_contour);


//        Mat mask = Mat::zeros(img_src.size(), CV_8UC1);
//        mask(rect0).setTo(255);
//        img_src.copyTo(img_src_hue_range, mask);

//        my_imshow("img_src_hue_range", img_src_hue_range);

//        if (img_src_hue_range_merged.empty()) {
//            img_src_hue_range_merged = img_src_hue_range;
//        } else {
//            addWeighted(img_src_hue_range_merged, 1.0, img_src_hue_range, 1.0, 0.0, img_src_hue_range_merged);
////            merge(img_src_hue_range_merged, img_src_hue_range);
//        }


        //img_src_hue_range_vector.push_back(img_src_hue_range);
    }

//    Mat img_src_hue_range_merged;
//    merge(img_src_hue_range_vector, img_src_hue_range_merged);
//
//    my_imshow("img_src_hue_range_merged", img_src_hue_range_merged);





//    Rect rect0 = boundingRect(Mat(contours_hue[1]));
//    cout << "包围盒测试" << endl;
//    cout << rect0 << endl;
//
//    Mat img_src_hue_range;
//    Mat mask = Mat::zeros(img_src_gray.size(), CV_8UC1);
//    mask(rect0).setTo(255);
//    img_src_gray.copyTo(img_src_hue_range, mask);
//    my_imshow("img_src_hue_range", img_src_hue_range);








//    cvtColor(img_src_hue_range_merged, img_src_gray, CV_BGR2GRAY);

//
//    // Canny Edge Detection
//    int lowThreshold = 55;
//    int ratio = 3;
//    int canny_kernal_size = 3;
//    Mat canny_detected_edges;
//
//    // Canny
//    CannyOperation(img_src, img_src_gray, img_dst, canny_detected_edges, lowThreshold, ratio, canny_kernal_size);
//
//    my_imshow("detect_edges", canny_detected_edges);
//    //my_imshow("dst", img_dst);
//
//    // Contour Finding and Drawing
//
//    vector<vector<Point> > contours;
//    vector<Vec4i> hierarchy;
//    Mat drawing_contour = ContourOperation(canny_detected_edges, contours, hierarchy, rng);
//
////    imwrite("/home/zhida/Desktop/example_data/contour_drawing.jpg", drawing_contour);
//
//    Mat img_src_with_contour;
//    img_src.copyTo(img_src_with_contour, drawing_contour);
//
//    my_imshow("drawing", drawing_contour);
////    my_imwrite("/home/zhida/Documents/Code/traffic-light-detection/resource/result/drawing.jpg", drawing_contour);
//////
//    my_imshow("img_src_with_contour", img_src_with_contour);
////    my_imwrite("/home/zhida/Documents/Code/traffic-light-detection/resource/result/img_src_with_contour.jpg", img_src_with_contour);
//
//    //Mat my_dst;
//    //addWeighted(img_src, 0.5, img_src_with_contour, 0.5, 20, my_dst);
//
//    //my_imshow("my_dst", my_dst);

    cv::waitKey(0);

    return 0;
}



