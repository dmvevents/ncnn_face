#ifndef TARCFACE_H
#define TARCFACE_H

#include <cmath>
#include <vector>
#include <string>
#include "ncnn/net.h"
#include <opencv2/highgui.hpp>
//----------------------------------------------------------------------------------------
//
// Created by Anton Alexander 06/26/2022
//
//----------------------------------------------------------------------------------------
using namespace std;

class TArcFace {
private:
    ncnn::Net net;
    const int feature_dim = 128;
    cv::Mat Zscore(const cv::Mat &fc);
public:
    TArcFace(void);
    ~TArcFace(void);

    cv::Mat GetFeature(cv::Mat img);
};
#endif
