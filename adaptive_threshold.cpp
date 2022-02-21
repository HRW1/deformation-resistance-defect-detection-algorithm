#include <cmath>
#include "Macro.h"

using namespace std;
using namespace cv;

int cal_threshold(const Mat &gray_img){
    Scalar mean;
    Scalar stddev;
    meanStdDev( gray_img, mean, stddev );
    double u = THR * mean.val[0];
    u = u < 30 ? u:30;
    u = u > 12 ? u:12;
    return (int)u;
}

double cal_entropy(cla_ifo *ci, int K){
    double ans = 0;
    for(int i = 1; i < K; i++){
        ans -= ((double)ci[i].N / ci[0].N) * log((double)ci[i].N / ci[0].N);
    }
    return ans/ log(ci[0].N);
}

double cal_density(cla_ifo *ci, int K){
    double ans = 0;
    for(int i = 1; i < K; i++){
        ans += ((double)ci[i].N / (double)ci[i].area) * ((double)ci[i].N / (double)ci[0].N);
    }
    return ans;
}