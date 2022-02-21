#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include "Macro.h"

using namespace std;
using namespace cv;

extern Mat img;

void clustering(int x, int y, int k, Mat &label, int max_d){
    label.at<int>(x,y) = k;
    for(int i=0; i<=max_d && (x+i) < label.rows; i++){//右下
        for(int j=0; j<=max_d && (y+j) < label.cols; j++){
            if(i*i+j*j > max_d*max_d) break;//超过距离限制
            if(label.at<int>(x+i,y+j) == -1){
                if(img.at<uchar>(x+i,y+j) == 0)label.at<int>(x+i,y+j) = 0;
                else clustering(x+i,y+j,k,label,max_d);
            }
        }
    }
    for(int i=-1; -i<=max_d && (x+i) >= 0; i--){//右上
        for(int j=0; j<=max_d && (y+j) < label.cols; j++){
            if(i*i+j*j > max_d*max_d) break;//超过距离限制
            if(label.at<int>(x+i,y+j) == -1){
                if(img.at<uchar>(x+i,y+j) == 0)label.at<int>(x+i,y+j) = 0;
                else clustering(x+i,y+j,k,label,max_d);
            }
        }
    }
    for(int i=0; i<=max_d && (x+i) < label.rows; i++){//左下
        for(int j=-1; -j<=max_d && (y+j) >= 0; j--){
            if(i*i+j*j > max_d*max_d) break;//超过距离限制
            if(label.at<int>(x+i,y+j) == -1){
                if(img.at<uchar>(x+i,y+j) == 0)label.at<int>(x+i,y+j) = 0;
                else clustering(x+i,y+j,k,label,max_d);
            }
        }
    }
    for(int i=-1; -i<=max_d && (x+i) >= 0; i--){//左上
        for(int j=-1; -j<=max_d && (y+j) >= 0; j--){
            if(i*i+j*j > max_d*max_d) break;//超过距离限制
            if(label.at<int>(x+i,y+j) == -1){
                if(img.at<uchar>(x+i,y+j) == 0)label.at<int>(x+i,y+j) = 0;
                else clustering(x+i,y+j,k,label,max_d);
            }
        }
    }
}

int cluster(Mat img, Mat &label, int max_d){
    int k = 1;//类别编号
    for(int i=0; i<img.rows; i++){
        for(int j=0; j<img.cols; j++){
            if(label.at<int>(i,j) == -1){
                if(img.at<uchar>(i,j) == 0) label.at<int>(i,j) = 0;//原图对应位置灰度值为0，则标记0
                else{
                    clustering(i, j, k, label, max_d);
                    k++;
                }
            }
        }
    }
    return k;
}