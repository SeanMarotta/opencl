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
    /* ----------------------------------------------------------------------------
                                    OPENCL PART
     ---------------------------------------------------------------------------- */
    // Parametres de Setup OpenCL
    cl_int err;
    cl_platform_id platform;
    cl_platform_id platforms[2];
    cl_context context;
    cl_device_id device_id; // FPGA ou GPU
    cl_command_queue command_queue;
    cl_program program;
    cl_kernel kernel;
    cl_event kernel_event;
    cl_ulong t_start = 0, t_end = 0;

    float *host_im, *host_filter;
    const string filename = "/home/imvia/CLionProjects/filter_2d_optimized/lena30.jpg";

    Mat img_grayscale = imread(filename, IMREAD_GRAYSCALE);
    const size_t N_image = img_grayscale.size().width;
    float* filter_name = averaging15x15;
    const int N_filter = 15;
    const int loc_wg_size = 32;

    // 1) Initialisation de la plateforme ------------------------------------------------------------------------------------------------------------------
    err = clGetPlatformIDs(2, platforms, nullptr);
    if (err != CL_SUCCESS)
    {
        printf("Creation plateform failed %d\n", err);
    }
    platform = platforms[0];

    // 2) Détection des Devices ------------------------------------------------------------------------------------------------------------------
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device_id, nullptr);
    if (err != CL_SUCCESS)
    {
        printf("Creation Device failed %d\n", err);
    }

    // 3) Création du contexte ------------------------------------------------------------------------------------------------------------------
    context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &err);
    if (!context)
    {
        printf("Creation du context failed\n");
        return EXIT_FAILURE;
    }

    // 4) Création du Command Queue ------------------------------------------------------------------------------------------------------------------
    command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
    if (!command_queue)
    {
        printf("Creation command queue failed\n");
        return EXIT_FAILURE;
    }

    // 5) Création des programmes ------------------------------------------------------------------------------------------------------------------
    program = clCreateProgramWithSource(context, 1, (const char**)&KernelSource, nullptr, &err);
    if (!program)
    {
        printf("Creation des programmes de traitement failed\n");
        return EXIT_FAILURE;
    }

    // 6) Compilation des programmes ------------------------------------------------------------------------------------------------------------------
    char compilerOptions[1024];
    sprintf(compilerOptions, "-D IMAGE_W=%d -D IMAGE_H=%d -D FILTER_SIZE=%d -D HALF_FILTER_SIZE=%d -D TWICE_HALF_FILTER_SIZE=%d -D HALF_FILTER_SIZE_IMAGE_W=%d",
            N_image,
            N_image,
            N_filter,
            N_filter/2,
            (N_filter/2) * 2,
            (N_filter/2) * N_image
    );
    err = clBuildProgram(program, 0, nullptr, compilerOptions, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Build program failed\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }

    // 7) Création des kernels ------------------------------------------------------------------------------------------------------------------
    kernel = clCreateKernel(program, "convolute", &err);
    if (!kernel || err != CL_SUCCESS)
    {
        printf("Creation du kernel failed\n");
        exit(1);
    }

    host_im = (float*)malloc(N_image * N_image * sizeof(float));
    for (int i = 0; i < N_image; i++)
    {
        for (int j = 0; j < N_image; j++)
        {
            host_im[i + j * N_image] = img_grayscale.at<uchar>(i, j);
        }
    }

    host_filter = (float*)malloc(N_filter * N_filter * sizeof(float));
    for (int i = 0; i < N_filter * N_filter; i++)
    {
        host_filter[i] = filter_name[i]; // Mettre le facteur
    }

    cl_mem a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N_image * N_image  * sizeof(float), host_im, &err);
    cl_mem b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N_filter * N_filter  * sizeof(float), host_filter, &err);
    cl_mem c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N_image * N_image * sizeof(float), nullptr, &err);
    delete [] host_im;

    size_t globalWorkSize[2] = { N_image, N_image };
    size_t localWorkSize[2] = { loc_wg_size,loc_wg_size};

    err = 0;
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &a);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &b);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &c);
    cout << ((localWorkSize[0] + 2 * (N_filter/2)) * (localWorkSize[1] + 2 * (N_filter/2)) * 4) << endl;
    err |= clSetKernelArg(kernel, 3, ((localWorkSize[0] + 2 * (N_filter/2)) * (localWorkSize[1] + 2 * (N_filter/2))*4), nullptr);
    //err |= clSetKernelArg(kernel, 3, 6397, nullptr); //(pour un filtre 5x5 mettre au moins 509 et 253 minimum pour un 3x3)
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }

    char string[1024];
    size_t size;
    size_t size_array[3];
    cl_ulong ulong;
    clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, 1024, string, nullptr);
    printf("\tCL_KERNEL_FUNCTION_NAME: %s\n\n", string);
    clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size), &size, nullptr);
    cout << "CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE : " << size << endl;
    clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size), &size, nullptr);
    cout << "CL_KERNEL_WORK_GROUP_SIZE : " << size << endl;
    clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof(size_array), &size_array, nullptr);
    cout << size_array[0] << ", "<< size_array[1] <<", " << size_array[2] << endl;
    clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(ulong), &ulong, nullptr);
    cout << "CL_KERNEL_LOCAL_MEM_SIZE : " << ulong << endl;


    // 12) Mise en route du système OpenCL (plusieurs itération pour obtenir le temps moyen de calcul effectué par le GPU) ---------------------------
    err = clEnqueueNDRangeKernel(command_queue, kernel, 2, nullptr, globalWorkSize, localWorkSize, 0, nullptr, &kernel_event);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to execute kernel! %d\n", err);
        return EXIT_FAILURE;
    }
    err = clWaitForEvents(1, &kernel_event);
    err = clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &t_end, nullptr);
    err |= clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &t_start, nullptr);
    if (err != CL_SUCCESS) {
        printf("%d\n", err);
        //return EXIT_FAILURE;
    }
    printf("Profile execution time = %0.3lf ms.\n", (double)(t_end - t_start)*1e-6);

    float* host_output;
    host_output = (float*)malloc(N_image * N_image * sizeof(float));
    clEnqueueReadBuffer(command_queue, c, CL_TRUE, 0, N_image * N_image * sizeof(float), host_output, 0, nullptr, nullptr);

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

    // 14) Libération des espaces mémoires des deux côtés
    clReleaseEvent(kernel_event);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
    return 0;
}
