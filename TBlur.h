#ifndef TBLUR_H
#define TBLUR_H
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//----------------------------------------------------------------------------------------
//
// Created by Anton Alexander 06/26/2022
//
//----------------------------------------------------------------------------------------
class TBlur
{
private:
    int cx, cy;
    int Block;
protected:
public:
    TBlur();
    virtual ~TBlur();
    double Execute(cv::Mat &frame);
};
//----------------------------------------------------------------------------------------
#endif // TBLUR_H
