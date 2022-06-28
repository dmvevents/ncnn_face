#ifndef SCRFD_H
#define SCRFD_H

#include <stdio.h>
#include <vector>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "ncnn/net.h"
#include "struct.h"


class SCRFD
{
private:
    ncnn::Net scrfd;
    int img_w;
    int img_h;
    ncnn::Option option_;

protected:
public:

    SCRFD(bool UseVulkan);
    virtual ~SCRFD();
    bool has_kps;

    int detect_scrfd(const cv::Mat& image, std::vector<FaceObject> &Faces);
    void draw_faceobjects(const cv::Mat& bgr);
};

#endif // SCRFD_H
