#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace std;
using namespace cv;

void MatchingMethod(int, void*);
void CannyOperation(Mat& src, Mat& src_gray, Mat& dst, Mat& detect_edges, int lowThreshold, int ratio, int kernal_size);
void my_imshow(string name, Mat img);
Mat ContourOperation(Mat canny_output, vector<vector<Point> > contours, vector<Vec4i> hierarchy, RNG rng);

int main(){

//    Mat img_src = cv::imread("/home/zhida/Desktop/example_data/2018_0318_155903_898.mp4_20180329_174351.047.jpg");
//    Mat img_src = cv::imread("/home/zhida/Desktop/example_data/2018_0318_155903_898.mp4_20180329_174342.713.jpg");
    Mat img_src = cv::imread("/home/zhida/Desktop/example_data/786849496.jpg");

    Mat img_src_gray;
//    Mat temple = cv::imread("/home/zhida/Desktop/example_data/traffic_light_temple.jpg");
//
//    cv::namedWindow("img1", CV_WINDOW_AUTOSIZE);
//    imshow("img1", img);
//    cv::waitKey(2500);
//
//    cout << img.size() << "qq"<<endl;

    Mat img_dst;
    img_dst.create(img_src.size(), img_src.type());

    cvtColor(img_src, img_src_gray, CV_BGR2GRAY);

    namedWindow("src_gray", WINDOW_AUTOSIZE);
    //imshow("src_gray", img_src_gray);

    // Canny Edge Detection
    int lowThreshold = 15;
    int ratio = 3;
    int canny_kernal_size = 3;
    Mat canny_detected_edges;

    CannyOperation(img_src, img_src_gray, img_dst, canny_detected_edges, lowThreshold, ratio, canny_kernal_size);

    my_imshow("detect_edges", canny_detected_edges);
    //my_imshow("dst", img_dst);

    // Contour Finding and Drawing
    RNG rng(12345);
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    Mat drawing_contour = ContourOperation(canny_detected_edges, contours, hierarchy, rng);

    imwrite("/home/zhida/Desktop/example_data/contour_drawing.jpg", drawing_contour);

    Mat img_src_with_contour;
    img_src.copyTo(img_src_with_contour, drawing_contour);

    my_imshow("drawing", drawing_contour);
    my_imshow("img_src_with_contour", img_src_with_contour);

    //Mat my_dst;
    //addWeighted(img_src, 0.5, img_src_with_contour, 0.5, 20, my_dst);

    //my_imshow("my_dst", my_dst);

    cv::waitKey(0);

    return 0;
}

Mat ContourOperation(Mat canny_output, vector<vector<Point> > contours, vector<Vec4i> hierarchy, RNG rng) {

    // Find coutours
    findContours(canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));

    // Draw contours
    Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
    for(int i=0; i<contours.size(); i++) {
        Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
    }

    return drawing;
}


void CannyOperation(Mat& src, Mat& src_gray, Mat& dst, Mat& detect_edges, int lowThreshold, int ratio, int kernal_size){

    blur(src_gray, detect_edges, Size(3,3));

    Canny(detect_edges, detect_edges, lowThreshold, lowThreshold*ratio, kernal_size);

    //dst = Scalar::all(0);

    src.copyTo(dst, detect_edges);

}

void my_imshow(string name, Mat img){

    namedWindow(name, WINDOW_AUTOSIZE);
    imshow(name, img);
}

void MatchingMethod(int, void*){
}


