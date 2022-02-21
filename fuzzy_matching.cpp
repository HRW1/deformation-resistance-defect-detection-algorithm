//
// 函数名：sub_img; 参数: img1待检测图, img2模板图, ans处理结果
// 方法: 模糊匹配,处理彩色图，处理结果为彩色图。
//
#include "string"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include "Macro.h"

using namespace std;
using namespace cv;

void sub_img(Mat img1, Mat img2, Mat &ans){
    uint min, temp, t1, t2, t3;
    for(int i = 0; i < img1.rows; i++){
        for(int j = 0; j < img1.cols; j++){
            min=10000;
            for(int x=-DX; x<=DX; x++){
                for(int y=-DY; y<=DY; y++){
                    if(i+x>=0 && i+x<img1.rows && j+y>=0 && j+y<img1.cols){
                        t1 = abs(img1.at<Vec3b>(i,j)[0] - img2.at<Vec3b>(i+x,j+y)[0]);
                        t2 = abs(img1.at<Vec3b>(i,j)[1] - img2.at<Vec3b>(i+x,j+y)[1]);
                        t3 = abs(img1.at<Vec3b>(i,j)[2] - img2.at<Vec3b>(i+x,j+y)[2]);
                        temp = t1 + t2 + t3;

                        if(temp < min){
                            min = temp;
                            ans.at<Vec3b>(i,j)[0] = t1;
                            ans.at<Vec3b>(i,j)[1] = t2;
                            ans.at<Vec3b>(i,j)[2] = t3;
                        }
                    }
                }
            }
        }
    }
}