#ifndef DEF_DET_MACRO_H
#define DEF_DET_MACRO_H

#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>

using namespace cv;

#define TIMES 3 //对原图的缩放比例
#define DX 3 //x方向模糊系数
#define DY 3 //y方向模糊系数
#define THR 20 //二值化阈值比平均灰度值
#define alph 0.32 //瑕疵密度在评价函数中所占比重
#define V 20 //体积滤波系数

struct cla_ifo{//瑕疵信息
    int xm=100000,xM=0,ym=100000,yM=0;//瑕疵大致范围
    int area=0,N=0;//瑕疵所占面积和瑕疵点数量
    bool is_def = false;//疑似瑕疵集合是否是瑕疵
};

void sub_img(Mat img1, Mat img2, Mat &ans);//模糊匹配
int cal_threshold(const Mat &gray_img);//根据模糊匹配结果获取阈值
int cluster(Mat img, Mat &label, int max_d);//根据二值化显著性图像聚类，获得label和距离阈值
void scan(Mat label, cla_ifo *ci, int max_d);//扫描label，获取cluster information数组
void paint(Mat &img, cla_ifo ci[], int k);//画出瑕疵大致范围
double cal_entropy(cla_ifo *ci, int K);//计算熵
double cal_density(cla_ifo *ci, int K);//计算瑕疵密度
void vol_filter(const Mat &gray_image, const Mat &label, int K, int thr, cla_ifo *ci);//体积滤波
void mask(const Mat &label, Mat &Mask, cla_ifo *ci, int max_d);//生成Mask

#endif //DEF_DET_MACRO_H
