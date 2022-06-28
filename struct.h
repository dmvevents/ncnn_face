#ifndef STRUCT_H
#define STRUCT_H

struct FaceObject
{
    cv::Rect_<float> rect;
    cv::Point2f landmark[5];
    int NameIndex;
    float FaceProb;
    double NameProb;
    double LiveProb;
    double Angle;
    int Color;      //background color of label on screen
};

#endif // STRUCT_H
