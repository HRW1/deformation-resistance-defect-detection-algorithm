#include "Macro.h"
#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int blocks[1000][2100];//存放对应的块所属的类别

void scan(Mat label, cla_ifo *ci, int max_d){
    int d;
    if(max_d <= 1) d = 1;
    else d = ceil((double)max_d * 1.4142 / 2);
    memset(blocks,0,sizeof blocks);
    for(int i=0; i<label.rows; i++){
        for(int j=0; j<label.cols; j++){
            int lb=label.at<int>(i,j);
            if(lb != 0){
                //更新方框位置
                ci[lb].xm=ci[lb].xm < i?ci[lb].xm:i;
                ci[lb].xM=ci[lb].xM > i?ci[lb].xM:i;
                ci[lb].ym=ci[lb].ym < j?ci[lb].ym:j;
                ci[lb].yM=ci[lb].yM > j?ci[lb].yM:j;
                //更新每个类别点数
                ci[lb].N++;
                ci[0].N++;
                //更新方框所属类别
                if(blocks[i / d][j / d] == 0) blocks[i / d][j / d] = lb;
            }
        }
    }
    //计算面积
    for(int i = 0; i < label.rows / d + 1; i++){
        for(int j = 0; j < label.cols / d + 1; j++){
            int w = d < (label.rows - i*d) ? d : (label.rows - i*d);
            int h = d < (label.cols - j*d) ? d : (label.cols - j*d);
            ci[blocks[i][j]].area += w * h;
        }
    }
}

void vol_filter(const Mat &gray_image, const Mat &label, int K, int thr, cla_ifo *ci){
    int sum[K];
    memset(sum,0,sizeof sum);
    for(int i=0; i<label.rows; i++){
        for(int j=0; j<label.cols; j++){
            int lb=label.at<int>(i,j);
            if(lb) sum[lb] += (int)gray_image.at<uchar>(i,j);
        }
    }
    for(int i=1; i<K; i++){
        if(sum[i] > V * thr) ci[i].is_def = true;
//        cout<<sum[i]<<endl;
    }
}

void mask(const Mat &label, Mat &Mask, cla_ifo *ci, int max_d){//Mask的值为0表示不是瑕疵，否则是瑕疵。
    for(int x = 0; x < label.rows; x++){
        for(int y = 0; y < label.cols; y++){
            for(int i=0; i<=max_d && (x+i) < label.rows; i++){//右下
                if(Mask.at<uchar>(x,y) == 255) break;//已经对(x,y)设置Mask
                for(int j=0; j<=max_d && (y+j) < label.cols; j++) {
                    //cout<<1<<" "<<x+i<<" "<<y+j<<endl;
                    if(i*i+j*j > max_d*max_d) break;//超过距离限制
                    int ind = label.at<int>(x+i,y+j);
                    if(ci[ind].is_def){
                        Mask.at<uchar>(x,y) = 255;
                        break;
                    }
                }
            }
            for(int i=-1; -i<=max_d && (x+i) >= 0; i--){//右上
                if(Mask.at<uchar>(x,y) == 255) break;//已经对(x,y)设置Mask
                for(int j=0; j<=max_d && (y+j) < label.cols; j++){
                    //cout<<2<<" "<<x+i<<" "<<y+j<<endl;
                    if(i*i+j*j > max_d*max_d) break;//超过距离限制
                    int ind = label.at<int>(x+i,y+j);
                    if(ci[ind].is_def){
                        Mask.at<uchar>(x,y) = 255;
                        break;
                    }
                }
            }
            for(int i=0; i<=max_d && (x+i) < label.rows; i++){//左下
                if(Mask.at<uchar>(x,y) == 255) break;//已经对(x,y)设置Mask
                for(int j=-1; -j<=max_d && (y+j) >= 0; j--){
                    //cout<<3<<" "<<x+i<<" "<<y+j<<endl;
                    if(i*i+j*j > max_d*max_d) break;//超过距离限制
                    int ind = label.at<int>(x+i,y+j);
                    //cout<< ind<<endl;
                    if(ci[ind].is_def){
                        Mask.at<uchar>(x,y) = 255;
                        break;
                    }
                }
            }
            for(int i=-1; -i<=max_d && (x+i) >= 0; i--){//左上
                if(Mask.at<uchar>(x,y) == 255) break;//已经对(x,y)设置Mask
                for(int j=-1; -j<=max_d && (y+j) >= 0; j--){
                    //cout<<4<<" "<<x+i<<" "<<y+j<<endl;
                    if(i*i+j*j > max_d*max_d) break;//超过距离限制
                    int ind = label.at<int>(x+i,y+j);
                    if(ci[ind].is_def){
                        Mask.at<uchar>(x,y) = 255;
                        break;
                    }
                }
            }
        }
    }
}

void paint(Mat &img, cla_ifo ci[], int k){
    for(int i=1; i < k; i++){
        if(!ci[i].is_def)continue;
        //cout<<ci[i].xm<<","<<ci[i].xM<<","<<ci[i].ym<<","<<ci[i].yM<<endl;
        line(img,Point(ci[i].yM,ci[i].xm),Point(ci[i].ym,ci[i].xm),Scalar( 0, 0, 255), 1, 8, 0);
        line(img,Point(ci[i].yM,ci[i].xm),Point(ci[i].yM,ci[i].xM),Scalar( 0, 0, 255), 1, 8, 0);
        line(img,Point(ci[i].ym,ci[i].xm),Point(ci[i].ym,ci[i].xM),Scalar( 0, 0, 255), 1, 8, 0);
        line(img,Point(ci[i].yM,ci[i].xM),Point(ci[i].ym,ci[i].xM),Scalar( 0, 0, 255), 1, 8, 0);
    }
}