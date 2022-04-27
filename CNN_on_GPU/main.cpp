#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include "CL/cl.h"
#include "squeezenet_params.h"
#include "kernel_source.h"

//#include "img/dog.h"
//#include "img/cat.h"
//#include "img/mouette.h"
//#include "img/dog_shiba.h"
//#include "img/panthere.h"
//#include "img/libellule.h"
//#include "img/gorille.h"
//#include "img/guitariste.h"
//#include "img/oiseau.h"
//#include "img/tigre.h"
//#include "img/motard.h"
//#include "img/chaise.h"
//#include "img/drapeau.h"
//#include "img/personnes.h"
//#include "img/panda_roux.h"
//#include "img/dormant.h"
//#include "img/poisson.h"
//#include "img/polaroid.h"
#include "img/rotisserie.h"
//#include "img/cheval.h"
//#include "img/phone.h"
//#include "img/trilobite.h"
//#include "img/tinca.h"


void* sample = sample_rotisserie;

// OpenCL runtime configuration
cl_int err;
cl_platform_id platforms[2];
cl_platform_id platform;
unsigned num_devices = 0;
cl_device_id device_id; // num_devices elements
cl_context context;
cl_command_queue queue;
cl_program program;

//host buffer
float* h_result_classifier = (float*)malloc((1000) * sizeof(float));
char class_label[201];

cl_kernel conv3x3, conv1x1, maxpool, avgpool;
cl_event conv3x3_evt, conv1x1_evt, maxpool_evt, avgpool_evt;

cl_mem d_sample, d_conv1_weight, d_conv1_bias, d_result_conv;
cl_mem d_result_pool1, d_result_block1_squeeze, d_result_block1_expand;
cl_mem d_fire1_squeeze_weight, d_fire1_squeeze_bias, d_fire1_expand1x1_weight, \
d_fire1_expand1x1_bias, d_fire1_expand3x3_weight, d_fire1_expand3x3_bias;
cl_mem d_fire2_squeeze_weight, d_fire2_squeeze_bias, d_fire2_expand1x1_weight, \
d_fire2_expand1x1_bias, d_fire2_expand3x3_weight, d_fire2_expand3x3_bias;
cl_mem d_result_block2_squeeze, d_result_block2_expand, d_result_pool2;
cl_mem d_fire3_squeeze_weight, d_fire3_squeeze_bias, d_fire3_expand1x1_weight, \
d_fire3_expand1x1_bias, d_fire3_expand3x3_weight, d_fire3_expand3x3_bias;
cl_mem d_fire4_squeeze_weight, d_fire4_squeeze_bias, d_fire4_expand1x1_weight, \
d_fire4_expand1x1_bias, d_fire4_expand3x3_weight, d_fire4_expand3x3_bias;
cl_mem d_result_pool3, d_result_block3_squeeze1, d_result_block3_expand1, d_result_block3_squeeze2;
cl_mem d_fire5_squeeze_weight, d_fire5_squeeze_bias, d_fire5_expand1x1_weight, \
d_fire5_expand1x1_bias, d_fire5_expand3x3_weight, d_fire5_expand3x3_bias;
cl_mem d_fire6_squeeze_weight, d_fire6_squeeze_bias, d_fire6_expand1x1_weight, \
d_fire6_expand1x1_bias, d_fire6_expand3x3_weight, d_fire6_expand3x3_bias;
cl_mem d_fire7_squeeze_weight, d_fire7_squeeze_bias, d_fire7_expand1x1_weight, \
d_fire7_expand1x1_bias, d_fire7_expand3x3_weight, d_fire7_expand3x3_bias;
cl_mem d_fire8_squeeze_weight, d_fire8_squeeze_bias, d_fire8_expand1x1_weight, \
d_fire8_expand1x1_bias, d_fire8_expand3x3_weight, d_fire8_expand3x3_bias;
cl_mem d_result_block3_expand2, d_result_classifier_conv, d_classifier_conv_weight, d_classifier_conv_bias, d_result_classifier;

void getLabel(unsigned int class_index);
void cleanup();
cl_ulong getStartEndTime(cl_event event);

int main() {

    // Initialize OpenCL.
    cl_int status;

    printf("Initializing OpenCL\n");

    // Get the OpenCL platform.
    err = clGetPlatformIDs(2, platforms, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Creation plateform failed %d\n", err);
    }
    platform = platforms[0];

    // Query the available OpenCL device.
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Creation Device failed %d\n", err);
    }

    // Create the context.
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context)
    {
        printf("Creation du context failed\n");
        return EXIT_FAILURE;
    }

    // Create the program for all device. Use the first device as the
    // representative device (assuming all device are of the same type).
    program = clCreateProgramWithSource(context, 1, (const char**)&KernelSource, NULL, &err);
    if (!program)
    {
        printf("Creation des programmes de traitement failed\n");
        return EXIT_FAILURE;
    }

    // Build the program that was just created.
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Build program failed\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }

    queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
    if (!queue)
    {
        printf("Creation command queue failed\n");
        return EXIT_FAILURE;
    }

    // Kernel.
    const char* kernel1 = "conv2d3x3";
    conv3x3 = clCreateKernel(program, kernel1, &status);

    const char* kernel2 = "maxpool2d";
    maxpool = clCreateKernel(program, kernel2, &status);

    const char* kernel3 = "conv2d1x1";
    conv1x1 = clCreateKernel(program, kernel3, &status);

    const char* kernel4 = "avgpool2d";
    avgpool = clCreateKernel(program, kernel4, &status);

    /**************************************************************/
    /*                          conv1                             */
    /**************************************************************/
    //Creat device buffers
    //d_sample = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
    //    sizeof(float) * (1 * 3 * 224 * 224), sample_lena, &status);
    d_sample = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              sizeof(float) * (1 * 3 * 224 * 224), sample, &status);
    //, "Failed to create buffer d_sample");

    //conv1 params
    d_conv1_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(conv1_weight), conv1_weight, &status);
    //, "Failed to create buffer d_conv1_weight");
    d_conv1_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  sizeof(conv1_bias), conv1_bias, &status);
    //, "Failed to create buffer d_conv1_bias");

    //conv1 result
    d_result_conv = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                   sizeof(float) * (1 * 64 * 111 * 111), NULL, &status);
    //, "Failed to create buffer d_result_conv");
    d_result_pool1 = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                    sizeof(float) * (1 * 64 * 55 * 55), NULL, &status);
    //, "Failed to create buffer d_result_pool1");

    //fire1
    d_result_block1_squeeze = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                             sizeof(float) * (1 * 16 * 55 * 55), NULL, &status);
    //, "Failed to create buffer d_result_block1_squeeze");
    d_result_block1_expand = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                            sizeof(float) * (1 * 128 * 55 * 55), NULL, &status);
    //, "Failed to create buffer d_result_block1_expand");
    d_result_pool2 = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                    sizeof(float) * (1 * 128 * 27 * 27), NULL, &status);
    //, "Failed to create buffer d_result_pool2");

    d_fire1_squeeze_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            sizeof(fire1_squeeze_weight), fire1_squeeze_weight, &status);
    //, "Failed to create buffer d_fire1_squeeze_weight");
    d_fire1_squeeze_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          sizeof(fire1_squeeze_bias), fire1_squeeze_bias, &status);
    //, "Failed to create buffer d_fire1_squeeze_bias");
    d_fire1_expand1x1_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                              sizeof(fire1_expand1x1_weight), fire1_expand1x1_weight, &status);
    //, "Failed to create buffer d_fire1_expand1x1_weight");
    d_fire1_expand1x1_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            sizeof(fire1_expand1x1_bias), fire1_expand1x1_bias, &status);
    //, "Failed to create buffer d_fire1_expand1x1_bias");
    d_fire1_expand3x3_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                              sizeof(fire1_expand3x3_weight), fire1_expand3x3_weight, &status);
    //, "Failed to create buffer d_fire1_expand3x3_weight");
    d_fire1_expand3x3_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            sizeof(fire1_expand3x3_bias), fire1_expand3x3_bias, &status);
    //, "Failed to create buffer d_fire1_expand3x3_bias");

    //fire2
    d_fire2_squeeze_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            sizeof(fire2_squeeze_weight), fire2_squeeze_weight, &status);
    //, "Failed to create buffer d_fire2_squeeze_weight");
    d_fire2_squeeze_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          sizeof(fire2_squeeze_bias), fire2_squeeze_bias, &status);
    //, "Failed to create buffer d_fire2_squeeze_bias");
    d_fire2_expand1x1_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                              sizeof(fire2_expand1x1_weight), fire2_expand1x1_weight, &status);
    //, "Failed to create buffer d_fire2_expand1x1_weight");
    d_fire2_expand1x1_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            sizeof(fire2_expand1x1_bias), fire2_expand1x1_bias, &status);
    //, "Failed to create buffer d_fire2_expand1x1_bias");
    d_fire2_expand3x3_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                              sizeof(fire2_expand3x3_weight), fire2_expand3x3_weight, &status);
    //, "Failed to create buffer d_fire2_expand3x3_weight");
    d_fire2_expand3x3_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            sizeof(fire2_expand3x3_bias), fire2_expand3x3_bias, &status);
    //, "Failed to create buffer d_fire2_expand3x3_bias");

    //block2
    d_result_block2_squeeze = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                             sizeof(float) * (1 * 32 * 27 * 27), NULL, &status);
    //, "Failed to create buffer d_result_block2_squeeze");
    d_result_block2_expand = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                            sizeof(float) * (1 * 256 * 27 * 27), NULL, &status);
    //, "Failed to create buffer d_result_block2_expand");
    d_result_pool3 = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                    sizeof(float) * (1 * 256 * 13 * 13), NULL, &status);
    //, "Failed to create buffer d_result_pool3");

    //fire3
    d_fire3_squeeze_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            sizeof(fire3_squeeze_weight), fire3_squeeze_weight, &status);
    //, "Failed to create buffer d_fire3_squeeze_weight");
    d_fire3_squeeze_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          sizeof(fire3_squeeze_bias), fire3_squeeze_bias, &status);
    //, "Failed to create buffer d_fire3_squeeze_bias");
    d_fire3_expand1x1_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                              sizeof(fire3_expand1x1_weight), fire3_expand1x1_weight, &status);
    //, "Failed to create buffer d_fire3_expand1x1_weight");
    d_fire3_expand1x1_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            sizeof(fire3_expand1x1_bias), fire3_expand1x1_bias, &status);
    //, "Failed to create buffer d_fire3_expand1x1_bias");
    d_fire3_expand3x3_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                              sizeof(fire3_expand3x3_weight), fire3_expand3x3_weight, &status);
    //, "Failed to create buffer d_fire3_expand3x3_weight");
    d_fire3_expand3x3_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            sizeof(fire3_expand3x3_bias), fire3_expand3x3_bias, &status);
    //, "Failed to create buffer d_fire3_expand3x3_bias");

    //fire4
    d_fire4_squeeze_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            sizeof(fire4_squeeze_weight), fire4_squeeze_weight, &status);
    //, "Failed to create buffer d_fire4_squeeze_weight");
    d_fire4_squeeze_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          sizeof(fire4_squeeze_bias), fire4_squeeze_bias, &status);
    //, "Failed to create buffer d_fire4_squeeze_bias");
    d_fire4_expand1x1_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                              sizeof(fire4_expand1x1_weight), fire4_expand1x1_weight, &status);
    //, "Failed to create buffer d_fire4_expand1x1_weight");
    d_fire4_expand1x1_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            sizeof(fire4_expand1x1_bias), fire4_expand1x1_bias, &status);
    //, "Failed to create buffer d_fire4_expand1x1_bias");
    d_fire4_expand3x3_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                              sizeof(fire4_expand3x3_weight), fire4_expand3x3_weight, &status);
    //, "Failed to create buffer d_fire4_expand3x3_weight");
    d_fire4_expand3x3_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            sizeof(fire4_expand3x3_bias), fire4_expand3x3_bias, &status);
    //, "Failed to create buffer d_fire4_expand3x3_bias");

    d_result_block3_squeeze1 = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                              sizeof(float) * (1 * 48 * 13 * 13), NULL, &status);
    //, "Failed to create buffer d_result_block3_squeeze1");
    d_result_block3_expand1 = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                             sizeof(float) * (1 * 384 * 13 * 13), NULL, &status);
    //, "Failed to create buffer d_result_block3_expand1");
    d_result_block3_squeeze2 = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                              sizeof(float) * (1 * 64 * 13 * 13), NULL, &status);
    //, "Failed to create buffer d_result_block3_squeeze2");
    d_result_block3_expand2 = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                             sizeof(float) * (1 * 512 * 13 * 13), NULL, &status);
    //, "Failed to create buffer d_result_block3_expand2");

    //fire5
    d_fire5_squeeze_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            sizeof(fire5_squeeze_weight), fire5_squeeze_weight, &status);
    //, "Failed to create buffer d_fire5_squeeze_weight");
    d_fire5_squeeze_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          sizeof(fire5_squeeze_bias), fire5_squeeze_bias, &status);
    //, "Failed to create buffer d_fire5_squeeze_bias");
    d_fire5_expand1x1_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                              sizeof(fire5_expand1x1_weight), fire5_expand1x1_weight, &status);
    //, "Failed to create buffer d_fire5_expand1x1_weight");
    d_fire5_expand1x1_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            sizeof(fire5_expand1x1_bias), fire5_expand1x1_bias, &status);
    //, "Failed to create buffer d_fire5_expand1x1_bias");
    d_fire5_expand3x3_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                              sizeof(fire5_expand3x3_weight), fire5_expand3x3_weight, &status);
    //, "Failed to create buffer d_fire5_expand3x3_weight");
    d_fire5_expand3x3_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            sizeof(fire5_expand3x3_bias), fire5_expand3x3_bias, &status);
    //, "Failed to create buffer d_fire5_expand3x3_bias");

    //fire6
    d_fire6_squeeze_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            sizeof(fire6_squeeze_weight), fire6_squeeze_weight, &status);
    //, "Failed to create buffer d_fire6_squeeze_weight");
    d_fire6_squeeze_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          sizeof(fire6_squeeze_bias), fire6_squeeze_bias, &status);
    //, "Failed to create buffer d_fire6_squeeze_bias");
    d_fire6_expand1x1_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                              sizeof(fire6_expand1x1_weight), fire6_expand1x1_weight, &status);
    //, "Failed to create buffer d_fire6_expand1x1_weight");
    d_fire6_expand1x1_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            sizeof(fire6_expand1x1_bias), fire6_expand1x1_bias, &status);
    //, "Failed to create buffer d_fire6_expand1x1_bias");
    d_fire6_expand3x3_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                              sizeof(fire6_expand3x3_weight), fire6_expand3x3_weight, &status);
    //, "Failed to create buffer d_fire6_expand3x3_weight");
    d_fire6_expand3x3_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            sizeof(fire6_expand3x3_bias), fire6_expand3x3_bias, &status);
    //, "Failed to create buffer d_fire6_expand3x3_bias");

    //fire7
    d_fire7_squeeze_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            sizeof(fire7_squeeze_weight), fire7_squeeze_weight, &status);
    //, "Failed to create buffer d_fire7_squeeze_weight");
    d_fire7_squeeze_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          sizeof(fire7_squeeze_bias), fire7_squeeze_bias, &status);
    //, "Failed to create buffer d_fire7_squeeze_bias");
    d_fire7_expand1x1_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                              sizeof(fire7_expand1x1_weight), fire7_expand1x1_weight, &status);
    //, "Failed to create buffer d_fire7_expand1x1_weight");
    d_fire7_expand1x1_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            sizeof(fire7_expand1x1_bias), fire7_expand1x1_bias, &status);
    //, "Failed to create buffer d_fire7_expand1x1_bias");
    d_fire7_expand3x3_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                              sizeof(fire7_expand3x3_weight), fire7_expand3x3_weight, &status);
    //, "Failed to create buffer d_fire7_expand3x3_weight");
    d_fire7_expand3x3_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            sizeof(fire7_expand3x3_bias), fire7_expand3x3_bias, &status);
    //, "Failed to create buffer d_fire7_expand3x3_bias");

    //fire8
    d_fire8_squeeze_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            sizeof(fire8_squeeze_weight), fire8_squeeze_weight, &status);
    //, "Failed to create buffer d_fire8_squeeze_weight");
    d_fire8_squeeze_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          sizeof(fire8_squeeze_bias), fire8_squeeze_bias, &status);
    //, "Failed to create buffer d_fire8_squeeze_bias");
    d_fire8_expand1x1_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                              sizeof(fire8_expand1x1_weight), fire8_expand1x1_weight, &status);
    //, "Failed to create buffer d_fire8_expand1x1_weight");
    d_fire8_expand1x1_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            sizeof(fire8_expand1x1_bias), fire8_expand1x1_bias, &status);
    //, "Failed to create buffer d_fire8_expand1x1_bias");
    d_fire8_expand3x3_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                              sizeof(fire8_expand3x3_weight), fire8_expand3x3_weight, &status);
    //, "Failed to create buffer d_fire8_expand3x3_weight");
    d_fire8_expand3x3_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            sizeof(fire8_expand3x3_bias), fire8_expand3x3_bias, &status);
    //, "Failed to create buffer d_fire8_expand3x3_bias");

    //classifier
    d_classifier_conv_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                              sizeof(classifier_conv_weight), classifier_conv_weight, &status);
    //, "Failed to create buffer classifier_conv_weight");
    d_classifier_conv_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            sizeof(classifier_conv_bias), classifier_conv_bias, &status);
    //, "Failed to create buffer d_classifier_conv_bias");

    d_result_classifier_conv = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                              sizeof(float) * (1 * 1000 * 13 * 13), NULL, &status);
    //, "Failed to create buffer d_result_classifier_conv");
    d_result_classifier = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                         sizeof(float) * 1000, NULL, &status);
    //, "Failed to create buffer d_result_classifier");

    /**************************************************************/
    /*                          conv1                             */
    /**************************************************************/
    printf("\r\nSqueezeNet on GPU start:\r\n");
    printf("kernel version 1.3\r\n\n");
    auto total = 0.0;
    auto t1 = std::chrono::high_resolution_clock::now();

    unsigned int input_channel, input_size, pad, stride, start_channel, output_size;

    input_channel = 3;
    input_size = 224;
    pad = 0;
    stride = 2;
    start_channel = 0;
    output_size = 111;

    status |= clSetKernelArg(conv3x3, 0, sizeof(int), &(input_channel));
    status |= clSetKernelArg(conv3x3, 1, sizeof(int), &(input_size));
    status |= clSetKernelArg(conv3x3, 2, sizeof(int), &(pad));
    status |= clSetKernelArg(conv3x3, 3, sizeof(int), &(stride));
    status |= clSetKernelArg(conv3x3, 4, sizeof(int), &(start_channel));
    status |= clSetKernelArg(conv3x3, 5, sizeof(int), &(output_size));
    status |= clSetKernelArg(conv3x3, 6, sizeof(cl_mem), &d_sample);
    status |= clSetKernelArg(conv3x3, 7, sizeof(cl_mem), &d_conv1_weight);
    status |= clSetKernelArg(conv3x3, 8, sizeof(cl_mem), &d_conv1_bias);
    status |= clSetKernelArg(conv3x3, 9, sizeof(cl_mem), &d_result_conv);
    //, "Setting conv1: conv3x3 arguments");

    size_t global_f[2] = { 64,111 };
    status = clEnqueueNDRangeKernel(queue, conv3x3, 2, NULL, global_f, NULL, 0, NULL, &conv3x3_evt);
    clWaitForEvents(1, &conv3x3_evt);
    printf("conv3x3 time: %0.3f ms\n", getStartEndTime(conv3x3_evt) * 1e-6);
    //, "Enqueueing conv1: conv3x3");

    input_size = 111;
    output_size = 55;
    status |= clSetKernelArg(maxpool, 0, sizeof(int), &(input_size));
    status |= clSetKernelArg(maxpool, 1, sizeof(int), &(output_size));
    status |= clSetKernelArg(maxpool, 2, sizeof(cl_mem), &d_result_conv);
    status |= clSetKernelArg(maxpool, 3, sizeof(cl_mem), &d_result_pool1);
    //, "Setting maxpool1 arguments");

    size_t global = 64;
    status = clEnqueueNDRangeKernel(queue, maxpool, 1, NULL, &global, NULL, 0, NULL, &maxpool_evt);
    //, "Enqueueing maxpool1");

    status = clFinish(queue);
    //, "Wait for maxpool1 finish");
    clWaitForEvents(1, &maxpool_evt);
    printf("maxpool  time: %0.3f ms\n", getStartEndTime(maxpool_evt) * 1e-6);

    auto t2 = std::chrono::high_resolution_clock::now();
    auto int_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1);
    printf("\rCONV1 takes: %0.3lf ms ------------------------\n\n", int_ms.count()*1e-6);
    total += int_ms.count();
    t1 = t2;
    /**************************************************************/
    /*                         block1                             */
    /**************************************************************/
    //fire1
    input_channel = 64 / 4;
    input_size = 55;

    status |= clSetKernelArg(conv1x1, 0, sizeof(int), &(input_channel));
    status |= clSetKernelArg(conv1x1, 1, sizeof(int), &(input_size));
    status |= clSetKernelArg(conv1x1, 2, sizeof(cl_mem), &d_result_pool1);
    status |= clSetKernelArg(conv1x1, 3, sizeof(cl_mem), &d_fire1_squeeze_weight);
    status |= clSetKernelArg(conv1x1, 4, sizeof(cl_mem), &d_fire1_squeeze_bias);
    status |= clSetKernelArg(conv1x1, 5, sizeof(cl_mem), &d_result_block1_squeeze);
    //, "Setting fire1_squeeze arguments");

    global_f[0] = 16; global_f[1] = 55;
    status = clEnqueueNDRangeKernel(queue, conv1x1, 2, NULL, global_f, NULL, 0, NULL, &conv1x1_evt);
    clWaitForEvents(1, &conv1x1_evt);
    printf("conv1x1 time: %0.3f ms\n", getStartEndTime(conv1x1_evt) * 1e-6);
    //, "Enqueueing fire1_squeeze");

    input_channel = 16 / 4;
    input_size = 55;

    status |= clSetKernelArg(conv1x1, 0, sizeof(int), &(input_channel));
    status |= clSetKernelArg(conv1x1, 1, sizeof(int), &(input_size));
    status |= clSetKernelArg(conv1x1, 2, sizeof(cl_mem), &d_result_block1_squeeze);
    status |= clSetKernelArg(conv1x1, 3, sizeof(cl_mem), &d_fire1_expand1x1_weight);
    status |= clSetKernelArg(conv1x1, 4, sizeof(cl_mem), &d_fire1_expand1x1_bias);
    status |= clSetKernelArg(conv1x1, 5, sizeof(cl_mem), &d_result_block1_expand);
    //, "Setting fire1_expand1x1 arguments");

    global_f[0] = 64; global_f[1] = 55;
    status = clEnqueueNDRangeKernel(queue, conv1x1, 2, NULL, global_f, NULL, 0, NULL, &conv1x1_evt);
    clWaitForEvents(1, &conv1x1_evt);
    printf("conv1x1 time: %0.3f ms\n", getStartEndTime(conv1x1_evt) * 1e-6);
    //, "Enqueueing fire1_expand1x1");

    status = clFinish(queue);
    //, "Wait for block1 finish");

    input_channel = 16;
    input_size = 55;
    pad = 1;
    stride = 1;
    start_channel = 64;
    output_size = 55;

    status |= clSetKernelArg(conv3x3, 0, sizeof(int), &(input_channel));
    status |= clSetKernelArg(conv3x3, 1, sizeof(int), &(input_size));
    status |= clSetKernelArg(conv3x3, 2, sizeof(int), &(pad));
    status |= clSetKernelArg(conv3x3, 3, sizeof(int), &(stride));
    status |= clSetKernelArg(conv3x3, 4, sizeof(int), &(start_channel));
    status |= clSetKernelArg(conv3x3, 5, sizeof(int), &(output_size));
    status |= clSetKernelArg(conv3x3, 6, sizeof(cl_mem), &d_result_block1_squeeze);
    status |= clSetKernelArg(conv3x3, 7, sizeof(cl_mem), &d_fire1_expand3x3_weight);
    status |= clSetKernelArg(conv3x3, 8, sizeof(cl_mem), &d_fire1_expand3x3_bias);
    status |= clSetKernelArg(conv3x3, 9, sizeof(cl_mem), &d_result_block1_expand);
    //, "Setting fire1_expand3x3 arguments");

    global_f[0] = 64; global_f[1] = 55;
    status = clEnqueueNDRangeKernel(queue, conv3x3, 2, NULL, global_f, NULL, 0, NULL, &conv3x3_evt);
    clWaitForEvents(1, &conv3x3_evt);
    printf("conv3x3 time: %0.3f ms\n", getStartEndTime(conv3x3_evt) * 1e-6);
    //, "Enqueueing fire1_expand3x3");

    //fire2
    input_channel = 128 / 4;
    input_size = 55;

    status |= clSetKernelArg(conv1x1, 0, sizeof(int), &(input_channel));
    status |= clSetKernelArg(conv1x1, 1, sizeof(int), &(input_size));
    status |= clSetKernelArg(conv1x1, 2, sizeof(cl_mem), &d_result_block1_expand);
    status |= clSetKernelArg(conv1x1, 3, sizeof(cl_mem), &d_fire2_squeeze_weight);
    status |= clSetKernelArg(conv1x1, 4, sizeof(cl_mem), &d_fire2_squeeze_bias);
    status |= clSetKernelArg(conv1x1, 5, sizeof(cl_mem), &d_result_block1_squeeze);
    //, "Setting fire2_squeeze arguments");

    global_f[0] = 16; global_f[1] = 55;
    status = clEnqueueNDRangeKernel(queue, conv1x1, 2, NULL, global_f, NULL, 0, NULL, &conv1x1_evt);
    clWaitForEvents(1, &conv1x1_evt);
    printf("conv1x1 time: %0.3f ms\n", getStartEndTime(conv1x1_evt) * 1e-6);
    //, "Enqueueing fire2_squeeze");

    input_channel = 16 / 4;
    input_size = 55;

    status |= clSetKernelArg(conv1x1, 0, sizeof(int), &(input_channel));
    status |= clSetKernelArg(conv1x1, 1, sizeof(int), &(input_size));
    status |= clSetKernelArg(conv1x1, 2, sizeof(cl_mem), &d_result_block1_squeeze);
    status |= clSetKernelArg(conv1x1, 3, sizeof(cl_mem), &d_fire2_expand1x1_weight);
    status |= clSetKernelArg(conv1x1, 4, sizeof(cl_mem), &d_fire2_expand1x1_bias);
    status |= clSetKernelArg(conv1x1, 5, sizeof(cl_mem), &d_result_block1_expand);
    //, "Setting fire2_expand1x1 arguments");

    global_f[0] = 64; global_f[1] = 55;
    status = clEnqueueNDRangeKernel(queue, conv1x1, 2, NULL, global_f, NULL, 0, NULL, &conv1x1_evt);
    clWaitForEvents(1, &conv1x1_evt);
    printf("conv1x1 time: %0.3f ms\n", getStartEndTime(conv1x1_evt) * 1e-6);
    //, "Enqueueing fire2_expand1x1");

    input_channel = 16;
    input_size = 55;
    pad = 1;
    stride = 1;
    start_channel = 64;
    output_size = 55;

    status |= clSetKernelArg(conv3x3, 0, sizeof(int), &(input_channel));
    status |= clSetKernelArg(conv3x3, 1, sizeof(int), &(input_size));
    status |= clSetKernelArg(conv3x3, 2, sizeof(int), &(pad));
    status |= clSetKernelArg(conv3x3, 3, sizeof(int), &(stride));
    status |= clSetKernelArg(conv3x3, 4, sizeof(int), &(start_channel));
    status |= clSetKernelArg(conv3x3, 5, sizeof(int), &(output_size));
    status |= clSetKernelArg(conv3x3, 6, sizeof(cl_mem), &d_result_block1_squeeze);
    status |= clSetKernelArg(conv3x3, 7, sizeof(cl_mem), &d_fire2_expand3x3_weight);
    status |= clSetKernelArg(conv3x3, 8, sizeof(cl_mem), &d_fire2_expand3x3_bias);
    status |= clSetKernelArg(conv3x3, 9, sizeof(cl_mem), &d_result_block1_expand);
    //, "Setting fire2_expand3x3 arguments");

    global_f[0] = 64; global_f[1] = 55;
    status = clEnqueueNDRangeKernel(queue, conv3x3, 2, NULL, global_f, NULL, 0, NULL, &conv3x3_evt);
    clWaitForEvents(1, &conv3x3_evt);
    printf("conv3x3 time: %0.3f ms\n", getStartEndTime(conv3x3_evt) * 1e-6);
    //, "Enqueueing fire2_expand3x3");

    input_size = 55;
    output_size = 27;
    status |= clSetKernelArg(maxpool, 0, sizeof(int), &(input_size));
    status |= clSetKernelArg(maxpool, 1, sizeof(int), &(output_size));
    status |= clSetKernelArg(maxpool, 2, sizeof(cl_mem), &d_result_block1_expand);
    status |= clSetKernelArg(maxpool, 3, sizeof(cl_mem), &d_result_pool2);
    //, "Setting maxpool2 arguments");

    global = 128;
    status = clEnqueueNDRangeKernel(queue, maxpool, 1, NULL, &global, NULL, 0, NULL, &maxpool_evt);
    clWaitForEvents(1, &maxpool_evt);
    printf("maxpool time: %0.3f ms\n", getStartEndTime(maxpool_evt) * 1e-6);
    //, "Enqueueing block1: maxpool");

    status = clFinish(queue);
    //, "Wait for block1 finish");

    t2 = std::chrono::high_resolution_clock::now();
    int_ms = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    printf("BLOCK1 takes: %0.3lf ms ------------------------\r\n\n", int_ms.count() * 1e-6);
    total += int_ms.count();
    t1 = t2;
    /**************************************************************/
    /*                         block2                             */
    /**************************************************************/
    //fire3
    input_channel = 128 / 4;
    input_size = 27;

    status |= clSetKernelArg(conv1x1, 0, sizeof(int), &(input_channel));
    status |= clSetKernelArg(conv1x1, 1, sizeof(int), &(input_size));
    status |= clSetKernelArg(conv1x1, 2, sizeof(cl_mem), &d_result_pool2);
    status |= clSetKernelArg(conv1x1, 3, sizeof(cl_mem), &d_fire3_squeeze_weight);
    status |= clSetKernelArg(conv1x1, 4, sizeof(cl_mem), &d_fire3_squeeze_bias);
    status |= clSetKernelArg(conv1x1, 5, sizeof(cl_mem), &d_result_block2_squeeze);
    //, "Setting fire3_squeeze arguments");

    global_f[0] = 32; global_f[1] = 27;
    status = clEnqueueNDRangeKernel(queue, conv1x1, 2, NULL, global_f, NULL, 0, NULL, &conv1x1_evt);
    clWaitForEvents(1, &conv1x1_evt);
    printf("conv1x1 time: %0.3f ms\n", getStartEndTime(conv1x1_evt) * 1e-6);
    //, "Enqueueing fire3_squeeze");

    input_channel = 32 / 4;
    input_size = 27;

    status |= clSetKernelArg(conv1x1, 0, sizeof(int), &(input_channel));
    status |= clSetKernelArg(conv1x1, 1, sizeof(int), &(input_size));
    status |= clSetKernelArg(conv1x1, 2, sizeof(cl_mem), &d_result_block2_squeeze);
    status |= clSetKernelArg(conv1x1, 3, sizeof(cl_mem), &d_fire3_expand1x1_weight);
    status |= clSetKernelArg(conv1x1, 4, sizeof(cl_mem), &d_fire3_expand1x1_bias);
    status |= clSetKernelArg(conv1x1, 5, sizeof(cl_mem), &d_result_block2_expand);
    //, "Setting fire3_expand1x1 arguments");

    global_f[0] = 128; global_f[1] = 27;
    status = clEnqueueNDRangeKernel(queue, conv1x1, 2, NULL, global_f, NULL, 0, NULL, &conv1x1_evt);
    clWaitForEvents(1, &conv1x1_evt);
    printf("conv1x1 time: %0.3f ms\n", getStartEndTime(conv1x1_evt) * 1e-6);
    //, "Enqueueing fire3_expand1x1");

    input_channel = 32;
    input_size = 27;
    pad = 1;
    stride = 1;
    start_channel = 128;
    output_size = 27;

    status |= clSetKernelArg(conv3x3, 0, sizeof(int), &(input_channel));
    status |= clSetKernelArg(conv3x3, 1, sizeof(int), &(input_size));
    status |= clSetKernelArg(conv3x3, 2, sizeof(int), &(pad));
    status |= clSetKernelArg(conv3x3, 3, sizeof(int), &(stride));
    status |= clSetKernelArg(conv3x3, 4, sizeof(int), &(start_channel));
    status |= clSetKernelArg(conv3x3, 5, sizeof(int), &(output_size));
    status |= clSetKernelArg(conv3x3, 6, sizeof(cl_mem), &d_result_block2_squeeze);
    status |= clSetKernelArg(conv3x3, 7, sizeof(cl_mem), &d_fire3_expand3x3_weight);
    status |= clSetKernelArg(conv3x3, 8, sizeof(cl_mem), &d_fire3_expand3x3_bias);
    status |= clSetKernelArg(conv3x3, 9, sizeof(cl_mem), &d_result_block2_expand);
    //, "Setting fire3_expand3x3 arguments");

    global_f[0] = 128; global_f[1] = 27;
    status = clEnqueueNDRangeKernel(queue, conv3x3, 2, NULL, global_f, NULL, 0, NULL, &conv3x3_evt);
    clWaitForEvents(1, &conv3x3_evt);
    printf("conv3x3 time: %0.3f ms\n", getStartEndTime(conv3x3_evt) * 1e-6);
    //, "Enqueueing fire3_expand3x3");

    status = clFinish(queue);
    //, "Wait for fire3 finish");

    //fire4
    input_channel = 256 / 4;
    input_size = 27;

    status |= clSetKernelArg(conv1x1, 0, sizeof(int), &(input_channel));
    status |= clSetKernelArg(conv1x1, 1, sizeof(int), &(input_size));
    status |= clSetKernelArg(conv1x1, 2, sizeof(cl_mem), &d_result_block2_expand);
    status |= clSetKernelArg(conv1x1, 3, sizeof(cl_mem), &d_fire4_squeeze_weight);
    status |= clSetKernelArg(conv1x1, 4, sizeof(cl_mem), &d_fire4_squeeze_bias);
    status |= clSetKernelArg(conv1x1, 5, sizeof(cl_mem), &d_result_block2_squeeze);
    //, "Setting fire4_squeeze arguments");

    global_f[0] = 32; global_f[1] = 27;
    status = clEnqueueNDRangeKernel(queue, conv1x1, 2, NULL, global_f, NULL, 0, NULL, &conv1x1_evt);
    clWaitForEvents(1, &conv1x1_evt);
    printf("conv1x1 time: %0.3f ms\n", getStartEndTime(conv1x1_evt) * 1e-6);
    //, "Enqueueing fire4_squeeze");

    input_channel = 32 / 4;
    input_size = 27;

    status |= clSetKernelArg(conv1x1, 0, sizeof(int), &(input_channel));
    status |= clSetKernelArg(conv1x1, 1, sizeof(int), &(input_size));
    status |= clSetKernelArg(conv1x1, 2, sizeof(cl_mem), &d_result_block2_squeeze);
    status |= clSetKernelArg(conv1x1, 3, sizeof(cl_mem), &d_fire4_expand1x1_weight);
    status |= clSetKernelArg(conv1x1, 4, sizeof(cl_mem), &d_fire4_expand1x1_bias);
    status |= clSetKernelArg(conv1x1, 5, sizeof(cl_mem), &d_result_block2_expand);
    //, "Setting fire4_expand1x1 arguments");

    global_f[0] = 128; global_f[1] = 27;
    status = clEnqueueNDRangeKernel(queue, conv1x1, 2, NULL, global_f, NULL, 0, NULL, &conv1x1_evt);
    clWaitForEvents(1, &conv1x1_evt);
    printf("conv1x1 time: %0.3f ms\n", getStartEndTime(conv1x1_evt) * 1e-6);
    //, "Enqueueing fire4_expand1x1");

    input_channel = 32;
    input_size = 27;
    pad = 1;
    stride = 1;
    start_channel = 128;
    output_size = 27;

    status |= clSetKernelArg(conv3x3, 0, sizeof(int), &(input_channel));
    status |= clSetKernelArg(conv3x3, 1, sizeof(int), &(input_size));
    status |= clSetKernelArg(conv3x3, 2, sizeof(int), &(pad));
    status |= clSetKernelArg(conv3x3, 3, sizeof(int), &(stride));
    status |= clSetKernelArg(conv3x3, 4, sizeof(int), &(start_channel));
    status |= clSetKernelArg(conv3x3, 5, sizeof(int), &(output_size));
    status |= clSetKernelArg(conv3x3, 6, sizeof(cl_mem), &d_result_block2_squeeze);
    status |= clSetKernelArg(conv3x3, 7, sizeof(cl_mem), &d_fire4_expand3x3_weight);
    status |= clSetKernelArg(conv3x3, 8, sizeof(cl_mem), &d_fire4_expand3x3_bias);
    status |= clSetKernelArg(conv3x3, 9, sizeof(cl_mem), &d_result_block2_expand);
    //, "Setting fire4_expand3x3 arguments");

    global_f[0] = 128; global_f[1] = 27;
    status = clEnqueueNDRangeKernel(queue, conv3x3, 2, NULL, global_f, NULL, 0, NULL, &conv3x3_evt);
    clWaitForEvents(1, &conv3x3_evt);
    printf("conv3x3 time: %0.3f ms\n", getStartEndTime(conv3x3_evt) * 1e-6);
    //, "Enqueueing fire4_expand3x3");

    status = clFinish(queue);
    //, "Wait for fire4 finish");

    input_size = 27;
    output_size = 13;
    status |= clSetKernelArg(maxpool, 0, sizeof(int), &(input_size));
    status |= clSetKernelArg(maxpool, 1, sizeof(int), &(output_size));
    status |= clSetKernelArg(maxpool, 2, sizeof(cl_mem), &d_result_block2_expand);
    status |= clSetKernelArg(maxpool, 3, sizeof(cl_mem), &d_result_pool3);
    //, "Setting block2: maxpool arguments");

    global = 256;
    status = clEnqueueNDRangeKernel(queue, maxpool, 1, NULL, &global, NULL, 0, NULL, &maxpool_evt);
    clWaitForEvents(1, &maxpool_evt);
    printf("maxpool time: %0.3f ms\n", getStartEndTime(maxpool_evt) * 1e-6);
    //, "Enqueueing block2: maxpool");

    status = clFinish(queue);
    //, "Wait for block2 finish");

    t2 = std::chrono::high_resolution_clock::now();
    int_ms = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    printf("BLOCK2 takes: %0.3lf ms ------------------------\r\n\n", int_ms.count() * 1e-6);
    total += int_ms.count();
    t1 = t2;
    /**************************************************************/
    /*                         block3                             */
    /**************************************************************/
    //fire5
    input_channel = 256 / 4;
    input_size = 13;

    status |= clSetKernelArg(conv1x1, 0, sizeof(int), &(input_channel));
    status |= clSetKernelArg(conv1x1, 1, sizeof(int), &(input_size));
    status |= clSetKernelArg(conv1x1, 2, sizeof(cl_mem), &d_result_pool3);
    status |= clSetKernelArg(conv1x1, 3, sizeof(cl_mem), &d_fire5_squeeze_weight);
    status |= clSetKernelArg(conv1x1, 4, sizeof(cl_mem), &d_fire5_squeeze_bias);
    status |= clSetKernelArg(conv1x1, 5, sizeof(cl_mem), &d_result_block3_squeeze1);
    //, "Setting fire5_squeeze arguments");

    global_f[0] = 48; global_f[1] = 13;
    status = clEnqueueNDRangeKernel(queue, conv1x1, 2, NULL, global_f, NULL, 0, NULL, &conv1x1_evt);
    clWaitForEvents(1, &conv1x1_evt);
    printf("conv1x1 time: %0.3f ms\n", getStartEndTime(conv1x1_evt) * 1e-6);
    //, "Enqueueing fire5_squeeze");

    input_channel = 48 / 4;
    input_size = 13;

    status |= clSetKernelArg(conv1x1, 0, sizeof(int), &(input_channel));
    status |= clSetKernelArg(conv1x1, 1, sizeof(int), &(input_size));
    status |= clSetKernelArg(conv1x1, 2, sizeof(cl_mem), &d_result_block3_squeeze1);
    status |= clSetKernelArg(conv1x1, 3, sizeof(cl_mem), &d_fire5_expand1x1_weight);
    status |= clSetKernelArg(conv1x1, 4, sizeof(cl_mem), &d_fire5_expand1x1_bias);
    status |= clSetKernelArg(conv1x1, 5, sizeof(cl_mem), &d_result_block3_expand1);
    //, "Setting fire5_expand1x1 arguments");

    global_f[0] = 192; global_f[1] = 13;
    status = clEnqueueNDRangeKernel(queue, conv1x1, 2, NULL, global_f, NULL, 0, NULL, &conv1x1_evt);
    clWaitForEvents(1, &conv1x1_evt);
    printf("conv1x1 time: %0.3f ms\n", getStartEndTime(conv1x1_evt) * 1e-6);
    //, "Enqueueing fire5_expand1x1");

    input_channel = 48;
    input_size = 13;
    pad = 1;
    stride = 1;
    start_channel = 192;
    output_size = 13;

    status |= clSetKernelArg(conv3x3, 0, sizeof(int), &(input_channel));
    status |= clSetKernelArg(conv3x3, 1, sizeof(int), &(input_size));
    status |= clSetKernelArg(conv3x3, 2, sizeof(int), &(pad));
    status |= clSetKernelArg(conv3x3, 3, sizeof(int), &(stride));
    status |= clSetKernelArg(conv3x3, 4, sizeof(int), &(start_channel));
    status |= clSetKernelArg(conv3x3, 5, sizeof(int), &(output_size));
    status |= clSetKernelArg(conv3x3, 6, sizeof(cl_mem), &d_result_block3_squeeze1);
    status |= clSetKernelArg(conv3x3, 7, sizeof(cl_mem), &d_fire5_expand3x3_weight);
    status |= clSetKernelArg(conv3x3, 8, sizeof(cl_mem), &d_fire5_expand3x3_bias);
    status |= clSetKernelArg(conv3x3, 9, sizeof(cl_mem), &d_result_block3_expand1);
    //, "Setting fire5_expand3x3 arguments");

    global_f[0] = 192; global_f[1] = 13;
    status = clEnqueueNDRangeKernel(queue, conv3x3, 2, NULL, global_f, NULL, 0, NULL, &conv3x3_evt);
    clWaitForEvents(1, &conv3x3_evt);
    printf("conv3x3 time: %0.3f ms\n", getStartEndTime(conv3x3_evt) * 1e-6);
    //, "Enqueueing fire5_expand3x3");

    status = clFinish(queue);
    //, "Wait for fire5 finish");

    //fire6
    input_channel = 384 / 4;
    input_size = 13;

    status |= clSetKernelArg(conv1x1, 0, sizeof(int), &(input_channel));
    status |= clSetKernelArg(conv1x1, 1, sizeof(int), &(input_size));
    status |= clSetKernelArg(conv1x1, 2, sizeof(cl_mem), &d_result_block3_expand1);
    status |= clSetKernelArg(conv1x1, 3, sizeof(cl_mem), &d_fire6_squeeze_weight);
    status |= clSetKernelArg(conv1x1, 4, sizeof(cl_mem), &d_fire6_squeeze_bias);
    status |= clSetKernelArg(conv1x1, 5, sizeof(cl_mem), &d_result_block3_squeeze1);
    //, "Setting fire6_squeeze arguments");

    global_f[0] = 48; global_f[1] = 13;
    status = clEnqueueNDRangeKernel(queue, conv1x1, 2, NULL, global_f, NULL, 0, NULL, &conv1x1_evt);
    clWaitForEvents(1, &conv1x1_evt);
    printf("conv1x1 time: %0.3f ms\n", getStartEndTime(conv1x1_evt) * 1e-6);
    //, "Enqueueing fire6_squeeze");

    input_channel = 48 / 4;
    input_size = 13;

    status |= clSetKernelArg(conv1x1, 0, sizeof(int), &(input_channel));
    status |= clSetKernelArg(conv1x1, 1, sizeof(int), &(input_size));
    status |= clSetKernelArg(conv1x1, 2, sizeof(cl_mem), &d_result_block3_squeeze1);
    status |= clSetKernelArg(conv1x1, 3, sizeof(cl_mem), &d_fire6_expand1x1_weight);
    status |= clSetKernelArg(conv1x1, 4, sizeof(cl_mem), &d_fire6_expand1x1_bias);
    status |= clSetKernelArg(conv1x1, 5, sizeof(cl_mem), &d_result_block3_expand1);
    //, "Setting fire6_expand1x1 arguments");

    global_f[0] = 192; global_f[1] = 13;
    status = clEnqueueNDRangeKernel(queue, conv1x1, 2, NULL, global_f, NULL, 0, NULL, &conv1x1_evt);
    clWaitForEvents(1, &conv1x1_evt);
    printf("conv1x1 time: %0.3f ms\n", getStartEndTime(conv1x1_evt) * 1e-6);
    //, "Enqueueing fire6_expand1x1");

    input_channel = 48;
    input_size = 13;
    pad = 1;
    stride = 1;
    start_channel = 192;
    output_size = 13;

    status |= clSetKernelArg(conv3x3, 0, sizeof(int), &(input_channel));
    status |= clSetKernelArg(conv3x3, 1, sizeof(int), &(input_size));
    status |= clSetKernelArg(conv3x3, 2, sizeof(int), &(pad));
    status |= clSetKernelArg(conv3x3, 3, sizeof(int), &(stride));
    status |= clSetKernelArg(conv3x3, 4, sizeof(int), &(start_channel));
    status |= clSetKernelArg(conv3x3, 5, sizeof(int), &(output_size));
    status |= clSetKernelArg(conv3x3, 6, sizeof(cl_mem), &d_result_block3_squeeze1);
    status |= clSetKernelArg(conv3x3, 7, sizeof(cl_mem), &d_fire6_expand3x3_weight);
    status |= clSetKernelArg(conv3x3, 8, sizeof(cl_mem), &d_fire6_expand3x3_bias);
    status |= clSetKernelArg(conv3x3, 9, sizeof(cl_mem), &d_result_block3_expand1);
    //, "Setting fire6_expand3x3 arguments");

    global_f[0] = 192; global_f[1] = 13;
    status = clEnqueueNDRangeKernel(queue, conv3x3, 2, NULL, global_f, NULL, 0, NULL, &conv3x3_evt);
    clWaitForEvents(1, &conv3x3_evt);
    printf("conv3x3 time: %0.3f ms\n", getStartEndTime(conv3x3_evt) * 1e-6);
    //, "Enqueueing fire6_expand3x3");

    status = clFinish(queue);
    //, "Wait for fire6 finish");

    //fire7
    input_channel = 384 / 4;
    input_size = 13;

    status |= clSetKernelArg(conv1x1, 0, sizeof(int), &(input_channel));
    status |= clSetKernelArg(conv1x1, 1, sizeof(int), &(input_size));
    status |= clSetKernelArg(conv1x1, 2, sizeof(cl_mem), &d_result_block3_expand1);
    status |= clSetKernelArg(conv1x1, 3, sizeof(cl_mem), &d_fire7_squeeze_weight);
    status |= clSetKernelArg(conv1x1, 4, sizeof(cl_mem), &d_fire7_squeeze_bias);
    status |= clSetKernelArg(conv1x1, 5, sizeof(cl_mem), &d_result_block3_squeeze2);
    //, "Setting fire7_squeeze arguments");

    global_f[0] = 64; global_f[1] = 13;
    status = clEnqueueNDRangeKernel(queue, conv1x1, 2, NULL, global_f, NULL, 0, NULL, &conv1x1_evt);
    clWaitForEvents(1, &conv1x1_evt);
    printf("conv1x1 time: %0.3f ms\n", getStartEndTime(conv1x1_evt) * 1e-6);
    //, "Enqueueing fire7_squeeze");

    input_channel = 64 / 4;
    input_size = 13;

    status |= clSetKernelArg(conv1x1, 0, sizeof(int), &(input_channel));
    status |= clSetKernelArg(conv1x1, 1, sizeof(int), &(input_size));
    status |= clSetKernelArg(conv1x1, 2, sizeof(cl_mem), &d_result_block3_squeeze2);
    status |= clSetKernelArg(conv1x1, 3, sizeof(cl_mem), &d_fire7_expand1x1_weight);
    status |= clSetKernelArg(conv1x1, 4, sizeof(cl_mem), &d_fire7_expand1x1_bias);
    status |= clSetKernelArg(conv1x1, 5, sizeof(cl_mem), &d_result_block3_expand2);
    //, "Setting fire7_expand1x1 arguments");

    global_f[0] = 256; global_f[1] = 13;
    status = clEnqueueNDRangeKernel(queue, conv1x1, 2, NULL, global_f, NULL, 0, NULL, &conv1x1_evt);
    clWaitForEvents(1, &conv1x1_evt);
    printf("conv1x1 time: %0.3f ms\n", getStartEndTime(conv1x1_evt) * 1e-6);
    //, "Enqueueing fire7_expand1x1");

    input_channel = 64;
    input_size = 13;
    pad = 1;
    stride = 1;
    start_channel = 256;
    output_size = 13;

    status |= clSetKernelArg(conv3x3, 0, sizeof(int), &(input_channel));
    status |= clSetKernelArg(conv3x3, 1, sizeof(int), &(input_size));
    status |= clSetKernelArg(conv3x3, 2, sizeof(int), &(pad));
    status |= clSetKernelArg(conv3x3, 3, sizeof(int), &(stride));
    status |= clSetKernelArg(conv3x3, 4, sizeof(int), &(start_channel));
    status |= clSetKernelArg(conv3x3, 5, sizeof(int), &(output_size));
    status |= clSetKernelArg(conv3x3, 6, sizeof(cl_mem), &d_result_block3_squeeze2);
    status |= clSetKernelArg(conv3x3, 7, sizeof(cl_mem), &d_fire7_expand3x3_weight);
    status |= clSetKernelArg(conv3x3, 8, sizeof(cl_mem), &d_fire7_expand3x3_bias);
    status |= clSetKernelArg(conv3x3, 9, sizeof(cl_mem), &d_result_block3_expand2);
    //, "Setting fire7_expand3x3 arguments");

    global_f[0] = 256; global_f[1] = 13;
    status = clEnqueueNDRangeKernel(queue, conv3x3, 2, NULL, global_f, NULL, 0, NULL, &conv3x3_evt);
    clWaitForEvents(1, &conv3x3_evt);
    printf("conv3x3 time: %0.3f ms\n", getStartEndTime(conv3x3_evt) * 1e-6);
    //, "Enqueueing fire7_expand3x3");

    status = clFinish(queue);
    //, "Wait for fire7 finish");

    //fire8
    input_channel = 512 / 4;
    input_size = 13;

    status |= clSetKernelArg(conv1x1, 0, sizeof(int), &(input_channel));
    status |= clSetKernelArg(conv1x1, 1, sizeof(int), &(input_size));
    status |= clSetKernelArg(conv1x1, 2, sizeof(cl_mem), &d_result_block3_expand2);
    status |= clSetKernelArg(conv1x1, 3, sizeof(cl_mem), &d_fire8_squeeze_weight);
    status |= clSetKernelArg(conv1x1, 4, sizeof(cl_mem), &d_fire8_squeeze_bias);
    status |= clSetKernelArg(conv1x1, 5, sizeof(cl_mem), &d_result_block3_squeeze2);
    //, "Setting fire8_squeeze arguments");

    global_f[0] = 64; global_f[1] = 13;
    status = clEnqueueNDRangeKernel(queue, conv1x1, 2, NULL, global_f, NULL, 0, NULL, &conv1x1_evt);
    clWaitForEvents(1, &conv1x1_evt);
    printf("conv1x1 time: %0.3f ms\n", getStartEndTime(conv1x1_evt) * 1e-6);
    //, "Enqueueing fire8_squeeze");

    input_channel = 64 / 4;
    input_size = 13;

    status |= clSetKernelArg(conv1x1, 0, sizeof(int), &(input_channel));
    status |= clSetKernelArg(conv1x1, 1, sizeof(int), &(input_size));
    status |= clSetKernelArg(conv1x1, 2, sizeof(cl_mem), &d_result_block3_squeeze2);
    status |= clSetKernelArg(conv1x1, 3, sizeof(cl_mem), &d_fire8_expand1x1_weight);
    status |= clSetKernelArg(conv1x1, 4, sizeof(cl_mem), &d_fire8_expand1x1_bias);
    status |= clSetKernelArg(conv1x1, 5, sizeof(cl_mem), &d_result_block3_expand2);
    //, "Setting fire8_expand1x1 arguments");

    global_f[0] = 256; global_f[1] = 13;
    status = clEnqueueNDRangeKernel(queue, conv1x1, 2, NULL, global_f, NULL, 0, NULL, &conv1x1_evt);
    clWaitForEvents(1, &conv1x1_evt);
    printf("conv1x1 time: %0.3f ms\n", getStartEndTime(conv1x1_evt) * 1e-6);
    //, "Enqueueing fire8_expand1x1");

    input_channel = 64;
    input_size = 13;
    pad = 1;
    stride = 1;
    start_channel = 256;
    output_size = 13;

    status |= clSetKernelArg(conv3x3, 0, sizeof(int), &(input_channel));
    status |= clSetKernelArg(conv3x3, 1, sizeof(int), &(input_size));
    status |= clSetKernelArg(conv3x3, 2, sizeof(int), &(pad));
    status |= clSetKernelArg(conv3x3, 3, sizeof(int), &(stride));
    status |= clSetKernelArg(conv3x3, 4, sizeof(int), &(start_channel));
    status |= clSetKernelArg(conv3x3, 5, sizeof(int), &(output_size));
    status |= clSetKernelArg(conv3x3, 6, sizeof(cl_mem), &d_result_block3_squeeze2);
    status |= clSetKernelArg(conv3x3, 7, sizeof(cl_mem), &d_fire8_expand3x3_weight);
    status |= clSetKernelArg(conv3x3, 8, sizeof(cl_mem), &d_fire8_expand3x3_bias);
    status |= clSetKernelArg(conv3x3, 9, sizeof(cl_mem), &d_result_block3_expand2);
    //, "Setting fire8_expand3x3 arguments");

    global_f[0] = 256; global_f[1] = 13;
    status = clEnqueueNDRangeKernel(queue, conv3x3, 2, NULL, global_f, NULL, 0, NULL, &conv3x3_evt);
    clWaitForEvents(1, &conv3x3_evt);
    printf("conv3x3 time: %0.3f ms\n", getStartEndTime(conv3x3_evt) * 1e-6);
    //, "Enqueueing fire8_expand3x3");

    status = clFinish(queue);
    //, "Wait for fire8 finish");

    t2 = std::chrono::high_resolution_clock::now();
    int_ms = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    printf("BLOCK3 takes: %0.3lf ms ------------------------\r\n\n", int_ms.count() * 1e-6);
    total += int_ms.count();
    t1 = t2;
    /**************************************************************/
    /*                       classifier                           */
    /**************************************************************/
    input_channel = 512 / 4;
    input_size = 13;

    status |= clSetKernelArg(conv1x1, 0, sizeof(int), &(input_channel));
    status |= clSetKernelArg(conv1x1, 1, sizeof(int), &(input_size));
    status |= clSetKernelArg(conv1x1, 2, sizeof(cl_mem), &d_result_block3_expand2);
    status |= clSetKernelArg(conv1x1, 3, sizeof(cl_mem), &d_classifier_conv_weight);
    status |= clSetKernelArg(conv1x1, 4, sizeof(cl_mem), &d_classifier_conv_bias);
    status |= clSetKernelArg(conv1x1, 5, sizeof(cl_mem), &d_result_classifier_conv);
    //, "Setting classifier_conv arguments");

    global_f[0] = 1000; global_f[1] = 13;
    status = clEnqueueNDRangeKernel(queue, conv1x1, 2, NULL, global_f, NULL, 0, NULL, &conv1x1_evt);
    clWaitForEvents(1, &conv1x1_evt);
    printf("conv1x1 time: %0.3f ms\n", getStartEndTime(conv1x1_evt) * 1e-6);
    //, "Enqueueing classifier_conv");

    status |= clSetKernelArg(avgpool, 0, sizeof(cl_mem), &d_result_classifier_conv);
    status |= clSetKernelArg(avgpool, 1, sizeof(cl_mem), &d_result_classifier);
    //, "Setting avgpool arguments");

    global = 1000;
    status = clEnqueueNDRangeKernel(queue, avgpool, 1, NULL, &global, NULL, 0, NULL, &avgpool_evt);
    clWaitForEvents(1, &avgpool_evt);
    printf("avgpool time: %0.3f ms\n", getStartEndTime(avgpool_evt) * 1e-6);
    //, "Enqueueing avgpool");

    status = clFinish(queue);
    //, "Wait for classifier finish");
    t2 = std::chrono::high_resolution_clock::now();
    status = clEnqueueReadBuffer(queue, d_result_classifier, CL_TRUE, 0, sizeof(float) * 1000, h_result_classifier, 0, NULL, NULL);

    float tmp = 0.0f;
    unsigned int class_index = 0;
    for (int j = 0; j < 1000; j++)
    {
        if (h_result_classifier[j] > tmp)
        {
            tmp = h_result_classifier[j];
            class_index = j;
        }
    }
    int_ms = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    printf("CLASSIFIER TAKES: %0.3lf ms ------------------------\r\n", int_ms.count() * 1e-6);
    total += int_ms.count();
    printf("\ntotal: %0.3lf ms\r\n", total * 1e-6);
    getLabel(class_index);
    printf("\r\npredicted label: %s\r\n", class_label);

    char string[1024];
    size_t size;

    clGetKernelInfo(conv1x1, CL_KERNEL_FUNCTION_NAME, 1024, string, NULL);
    printf("\tCL_KERNEL_FUNCTION_NAME: %s\n", string);
    clGetKernelInfo(conv3x3, CL_KERNEL_FUNCTION_NAME, 1024, string, NULL);
    printf("\tCL_KERNEL_FUNCTION_NAME: %s\n", string);
    clGetKernelInfo(maxpool, CL_KERNEL_FUNCTION_NAME, 1024, string, NULL);
    printf("\tCL_KERNEL_FUNCTION_NAME: %s\n", string);
    clGetKernelInfo(avgpool, CL_KERNEL_FUNCTION_NAME, 1024, string, NULL);
    printf("\tCL_KERNEL_FUNCTION_NAME: %s\n", string);

    cleanup();

    printf("done\n");

    return 0;
}

cl_ulong getStartEndTime(cl_event event) {
    cl_int status;

    cl_ulong start, end;
    status = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    status = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);

    return end - start;
}

void getLabel(unsigned int class_index)
{
    int i;

    FILE* fp;

    fp = fopen("/home/imvia/CLionProjects/CNN_on_GPU/synset_words.txt", "r");
    for (i = 0; i < class_index + 1; i++)
    {
        fgets(class_label, sizeof(class_label), fp);
    }
    fclose(fp);
}

void cleanup()
{

    clReleaseEvent(conv3x3_evt);
    clReleaseEvent(conv1x1_evt);
    clReleaseEvent(maxpool_evt);
    clReleaseEvent(avgpool_evt);

    clReleaseMemObject(d_sample);
    clReleaseMemObject(d_conv1_weight);
    clReleaseMemObject(d_conv1_bias);
    clReleaseMemObject(d_result_conv);

    clReleaseMemObject(d_result_pool1);
    clReleaseMemObject(d_result_block1_squeeze);
    clReleaseMemObject(d_result_block1_expand);

    clReleaseMemObject(d_fire1_squeeze_weight);
    clReleaseMemObject(d_fire1_squeeze_bias);
    clReleaseMemObject(d_fire1_expand1x1_weight);
    clReleaseMemObject(d_fire1_expand1x1_bias);
    clReleaseMemObject(d_fire1_expand3x3_weight);
    clReleaseMemObject(d_fire1_expand3x3_bias);

    clReleaseMemObject(d_fire2_squeeze_weight);
    clReleaseMemObject(d_fire2_squeeze_bias);
    clReleaseMemObject(d_fire2_expand1x1_weight);
    clReleaseMemObject(d_fire2_expand1x1_bias);
    clReleaseMemObject(d_fire2_expand3x3_weight);
    clReleaseMemObject(d_fire2_expand3x3_bias);

    clReleaseMemObject(d_result_block2_squeeze);
    clReleaseMemObject(d_result_block2_expand);
    clReleaseMemObject(d_result_pool2);

    clReleaseMemObject(d_fire3_squeeze_weight);
    clReleaseMemObject(d_fire3_squeeze_bias);
    clReleaseMemObject(d_fire3_expand1x1_weight);
    clReleaseMemObject(d_fire3_expand1x1_bias);
    clReleaseMemObject(d_fire3_expand3x3_weight);
    clReleaseMemObject(d_fire3_expand3x3_bias);

    clReleaseMemObject(d_fire4_squeeze_weight);
    clReleaseMemObject(d_fire4_squeeze_bias);
    clReleaseMemObject(d_fire4_expand1x1_weight);
    clReleaseMemObject(d_fire4_expand1x1_bias);
    clReleaseMemObject(d_fire4_expand3x3_weight);
    clReleaseMemObject(d_fire4_expand3x3_bias);

    clReleaseMemObject(d_result_pool3);
    clReleaseMemObject(d_result_block3_squeeze1);
    clReleaseMemObject(d_result_block3_expand1);
    clReleaseMemObject(d_result_block3_squeeze2);

    clReleaseMemObject(d_fire5_squeeze_weight);
    clReleaseMemObject(d_fire5_squeeze_bias);
    clReleaseMemObject(d_fire5_expand1x1_weight);
    clReleaseMemObject(d_fire5_expand1x1_bias);
    clReleaseMemObject(d_fire5_expand3x3_weight);
    clReleaseMemObject(d_fire5_expand3x3_bias);

    clReleaseMemObject(d_fire6_squeeze_weight);
    clReleaseMemObject(d_fire6_squeeze_bias);
    clReleaseMemObject(d_fire6_expand1x1_weight);
    clReleaseMemObject(d_fire6_expand1x1_bias);
    clReleaseMemObject(d_fire6_expand3x3_weight);
    clReleaseMemObject(d_fire6_expand3x3_bias);

    clReleaseMemObject(d_fire7_squeeze_weight);
    clReleaseMemObject(d_fire7_squeeze_bias);
    clReleaseMemObject(d_fire7_expand1x1_weight);
    clReleaseMemObject(d_fire7_expand1x1_bias);
    clReleaseMemObject(d_fire7_expand3x3_weight);
    clReleaseMemObject(d_fire7_expand3x3_bias);

    clReleaseMemObject(d_fire8_squeeze_weight);
    clReleaseMemObject(d_fire8_squeeze_bias);
    clReleaseMemObject(d_fire8_expand1x1_weight);
    clReleaseMemObject(d_fire8_expand1x1_bias);
    clReleaseMemObject(d_fire8_expand3x3_weight);
    clReleaseMemObject(d_fire8_expand3x3_bias);

    clReleaseMemObject(d_result_block3_expand2);
    clReleaseMemObject(d_result_classifier_conv);

    clReleaseMemObject(d_classifier_conv_weight);
    clReleaseMemObject(d_classifier_conv_bias);
    clReleaseMemObject(d_result_classifier);

    clReleaseKernel(conv3x3);
    clReleaseKernel(conv1x1);
    clReleaseKernel(maxpool);
    clReleaseKernel(avgpool);

    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);

    free(h_result_classifier);
}
