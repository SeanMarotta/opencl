#include <CL/cl.h>
#include <cstdlib>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include "test_out.h"
#include "filter_2d.h"
#include "kernel_source.h"

using namespace cv;
using namespace std;

int main() {
    float *host_im, *host_filter;
    const string filename = "/home/imvia/CLionProjects/filter_2d_optimized/lena30.jpg";

    Mat img_grayscale = imread(filename, IMREAD_COLOR);
    //cout << img_grayscale << endl;
    const size_t N_image = img_grayscale.size().width;
    float* filter_name = averaging15x15;
    const int N_filter = 15;
    const int loc_wg_size = 32;

    host_im = (float*)malloc(N_image * N_image * sizeof(float) * 3);
    for (int i = 0; i < N_image; i++)
    {
        for (int j = 0; j < N_image; j++)
        {
            for (int c = 0; c < 3 ; ++c)
            {
                host_im[i * N_image * N_image + j * N_image + c] = img_grayscale.at<uchar>(i, j);
                cout << host_im[i * N_image * N_image + j * N_image + c] << endl;
            }
        }
    }


    float* host_output;
    host_output = (float*)malloc(N_image * N_image * sizeof(float) * 4);


    // Affichage de l'image aprés traitement par le GPU -----------------------------------------------------------------------------------------
    Mat img_grayscale_f = cv::Mat(N_image, N_image, CV_32F);
    for (int i = 0; i < N_image; i++)
    {
        for (int j = 0; j < N_image; j++)
        {
            img_grayscale_f.at<float>(i, j) = host_output[i + j * N_image] / 255;
        }
    }
    imshow("grayscale image filtered", img_grayscale_f);
    imshow("grayscale image not filtered", img_grayscale);
    waitKey(0);
    destroyAllWindows();
    return 0;
}
