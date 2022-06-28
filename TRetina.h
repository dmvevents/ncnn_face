#ifndef TRETINA_H
#define TRETINA_H
#include <opencv2/highgui.hpp>
#include <stdio.h>
#include <vector>
//#include "net.h"
#include "ncnn/net.h"
#include "struct.h"

//----------------------------------------------------------------------------------------
//
// Created by ncnn
//
// Modified by Q-engineering 2020/12/28
//
//----------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------
class TRetina
{
private:
    ncnn::Net retinaface;
    int img_w;
    int img_h;
protected:
public:
    TRetina(int Width,int Height,bool UseVulkan);
    virtual ~TRetina();

    int detect_retinaface(const cv::Mat& bgr,std::vector<FaceObject> &Faces);
    void draw_faceobjects(const cv::Mat& bgr);
};

#endif // TRETINA_H
