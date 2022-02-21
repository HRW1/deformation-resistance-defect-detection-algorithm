#include <iostream>
#include <cmath>
#include "string"
#include "vector"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <ctime>
#include "Macro.h"

using namespace std;
using namespace cv;

int OP_vol = 1; //使用体积滤波OP_vol=1，使用形态学OP_vol=0
Mat img;
string s="6";

int main() {
    Mat img1,image1,imgt,imaget;
    Mat RdBoxRes, Mask;

    img1 = imread("./data_tc/"+s+".jpg");
    imgt = imread("./data_tc/"+s+"t.jpg");

//    img1 = imread("E:/Clion/def_det/deform/10.jpg");
//    imgt = imread("E:/Clion/def_det/deform/0.jpg");

    if(img1.rows != imgt.rows || img1.cols != imgt.cols){
        cout<<"Template size ("<<imgt.rows<<", "<<imgt.cols<<") is not equal to test image's ("<<img1.rows<<", "<<img1.cols<<")."<<endl;
        return 0;
    }

    //图像预处理：缩放
    resize(img1, image1, Size(img1.cols / TIMES, img1.rows / TIMES), 0, 0, INTER_LINEAR);
    resize(imgt, imaget, Size(imgt.cols / TIMES, imgt.rows / TIMES), 0, 0, INTER_LINEAR);

    //模糊匹配
    cout<<"Fuzzy matching start."<<endl;
    Mat sub(img1.rows / TIMES,img1.cols / TIMES, CV_8UC3, Scalar(0,0,0));
    clock_t start = clock();
    sub_img(image1,imaget,sub);
    clock_t end   = clock();
    cout<<"Fuzzy matching finished. Cost of time: "<<end - start<<"ms."<<endl;

    //显著性图像转灰度
    Mat sub_gray,mat_mean,mat_stddev;
    cvtColor(sub, sub_gray, COLOR_BGR2GRAY);
//    namedWindow("sub_gray", WINDOW_AUTOSIZE);
//    imshow("sub_gray", sub_gray);

//    imwrite("./100.jpg",sub_gray);

    //动态阈值二值化???
    int thr = cal_threshold(sub_gray);
    cout<<"threshold = "<<thr<<endl;
    threshold(sub_gray,img, thr,255,THRESH_BINARY);

    //后处理
    if(OP_vol) {
        //volume filter
        cout << "Calculating parameter MAX_D..." << endl;
        cout << "**************************************" << endl;
        int fin_d = 2;//距离阈值Eps
        double max_res = -10.0;
        start = clock();
        for (int i = 2; i < 40; i++) {//遍历 max_d
            //连通图法凝聚
            int max_d = i;
            Mat label(img.rows, img.cols, CV_32S, Scalar::all(-1));
            int classes = cluster(img, label, max_d);
            cla_ifo ci[classes];//ci[0]存储聚类的信息

            scan(label, ci, max_d);

            //计算熵
            double e = cal_entropy(ci, classes);

            //计算密度
            double rou = cal_density(ci, classes);

            if (max_res < (alph * rou - (1 - alph) * e)) {
                max_res = alph * rou - (1 - alph) * e;
                fin_d = i;
            }

            cout << "*";
        }
        end = clock();
        cout << endl << "Calculating parameter MAX_D finished. Cost of time: "<< (double) (end - start) / 39.0 <<"ms per loop." << endl;

        //获取瑕疵聚类信息
        Mat label(img.rows, img.cols, CV_32S, Scalar::all(-1));
        int classes = cluster(img, label, fin_d);
        cla_ifo ci[classes];//cluster information. ci[0]存储聚类的信息
        scan(label, ci, fin_d);
        cout << "ci[0].area=" << ci[0].area << " ci[0].N=" << ci[0].N << endl;
        cout << "classes = " << classes << " MAX_D = " << fin_d << endl;

        //体积滤波
        vol_filter(sub_gray, label, classes, thr, ci);

        //生成Mask
        Mask=Mat(img.rows, img.cols, CV_8U, Scalar::all(0));
        fin_d = fin_d > 10? 10 : fin_d;
        mask(label, Mask, ci, fin_d);

        //生成并显示红框标注
        RdBoxRes = image1;

        paint(RdBoxRes,ci,classes);
        namedWindow("sub", WINDOW_AUTOSIZE);
        imshow("sub", RdBoxRes);
    }else{
        //开运算
        Mat element;
        element = getStructuringElement(MORPH_RECT, Size(1, 1));
        morphologyEx(img, Mask, MORPH_OPEN, element);
    }

    //读取人工标注图
    Mat mark,mark1;
    mark = imread("./mark/"+s+".jpg");
    if(!mark.empty()) {
        resize(mark, mark1, Size(img1.cols / TIMES, img1.rows / TIMES), 0, 0, INTER_LINEAR);
        cvtColor(mark1, mark, COLOR_BGR2GRAY);

        //计算F1,p,r,iou...
        int TP = 0, TN = 0, FP = 0, FN = 0;
        Mat clr_res(mark.rows, mark.cols, CV_8UC3);
        for (int i = 0; i < mark.rows; i++) {
            for (int j = 0; j < mark.cols; j++) {
                uchar res = Mask.at<uchar>(i, j);
                uchar mrk = mark.at<uchar>(i, j);
                if (res > 0 && mrk > 0) {//TP
                    clr_res.at<Vec3b>(i, j)[0] = 0;//B
                    clr_res.at<Vec3b>(i, j)[1] = 255;//G
                    clr_res.at<Vec3b>(i, j)[2] = 0;//R
                    TP++;
                } else if (res > 0 && mrk == 0) {//FP
                    clr_res.at<Vec3b>(i, j)[0] = 255;
                    clr_res.at<Vec3b>(i, j)[1] = 0;
                    clr_res.at<Vec3b>(i, j)[2] = 0;
                    FP++;
                } else if (res == 0 && mrk > 0) {//FN
                    clr_res.at<Vec3b>(i, j)[0] = 0;
                    clr_res.at<Vec3b>(i, j)[1] = 0;
                    clr_res.at<Vec3b>(i, j)[2] = 255;
                    FN++;
                } else {//TN
                    clr_res.at<Vec3b>(i, j)[0] = 0;
                    clr_res.at<Vec3b>(i, j)[1] = 0;
                    clr_res.at<Vec3b>(i, j)[2] = 0;
                    TN++;
                }
            }
        }
        double TPR = (double) TP / (TP + FP);
        double FPR = (double) FP / (FP + TN);
        double PPV = (double) TP / (TP + FN);
        double NPV = (double) TN / (TN + FN);
        double F1 = 2 * TPR * PPV / (TPR + PPV);
//        double IOU = (double) TP / (TP + FP + FN);
        cout << "TPR=" << TPR  << "\tPPV=" << PPV  << "\tF1=" << F1 <<endl;
        cout <<setiosflags(ios::fixed) <<setprecision(1) << F1*100 << "/" << TPR*100  << "/" << PPV*100 <<endl;

        //输出彩色结果
//    imwrite("./res/"+s+"_6.jpg",clr_res);

        //显示彩色结果
        namedWindow("clr_res", WINDOW_AUTOSIZE);
        imshow("clr_res", clr_res);
    } else{
        cout<<"Marked image is not found!"<<endl;
    }

//    namedWindow("mask", WINDOW_AUTOSIZE);
//    imshow("mask", Mask);

    waitKey(0);
    return 0;
}
