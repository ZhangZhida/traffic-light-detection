#ifndef SELECTROIBYMOUSE_H
#define SELECTROIBYMOUSE_H

void mouseHandlerColor(int event, int x, int y, int flags, void *param);
void getColor(const cv::Mat& img, double scale,int mode);

#endif //SELECTROIBYMOUSE_H