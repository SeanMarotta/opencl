#include <CL/cl.h>
#include <cstdlib>
#include <cstdio>
#include <iostream>
using namespace std;

const char* KernelSource = "\n" \
"__kernel void                                        \n"\
"vadd( __global float *aq, __global float *bq, __global float *cq) {         \n"\
"   int i = get_global_id(0);                                       \n"\
"   cq[i] = aq[i] + bq[i];                                             \n"\
"}                                                                  \n"\
"\n";

int main() {
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

    // 1) Initialisation de la plateforme ------------------------------------------------------------------------------------------------------------------
    err = clGetPlatformIDs(2, platforms, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Creation plateform failed %d\n", err);
    }
    platform = platforms[0];

    // 2) Détection des Devices ------------------------------------------------------------------------------------------------------------------
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Creation Device failed %d\n", err);
    }

    // 3) Création du contexte ------------------------------------------------------------------------------------------------------------------
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
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
    program = clCreateProgramWithSource(context, 1, (const char**)&KernelSource, NULL, &err);
    if (!program)
    {
        printf("Creation des programmes de traitement failed\n");
        return EXIT_FAILURE;
    }

    // 6) Compilation des programmes ------------------------------------------------------------------------------------------------------------------
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

    // 7) Création des kernels ------------------------------------------------------------------------------------------------------------------
    kernel = clCreateKernel(program, "vadd", &err);
    if (!kernel || err != CL_SUCCESS)
    {
        printf("Creation du kernel failed\n");
        exit(1);
    }

    // 8) Création des buffers (côté Device) ------------------------------------------------------------------------------------------------------
    float *host_a, *host_b;
    host_a = (float*)malloc(10 * sizeof(*host_a));
    host_b = (float*)malloc(10 * sizeof(*host_b));
    for (int i = 0; i < 10; ++i) {
        host_a[i] = (float)i;
        host_b[i] = (float)i*i;
    }

    cl_mem a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 10 * sizeof(float), host_a, &err);
    cl_mem b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 10 * sizeof(float), host_b, &err);
    cl_mem c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 10 * sizeof(float), NULL, &err);

    // 10) Mappage des paramètres du kernel --------------------------------------------------------------------------------------------------------
    err = 0;
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &a);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &b);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &c);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }

    size_t globalWorkSize = 10;
    err = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &globalWorkSize, NULL, 0, NULL, &kernel_event);
    if (err)
    {
        printf("Error: Failed to execute kernel! %d\n", err);
        return EXIT_FAILURE;
    }
    float* host_c;
    host_c = (float*)malloc(10 * sizeof(float));
    clEnqueueReadBuffer(command_queue, c, CL_TRUE, 0, 10 * sizeof(float), host_c, 0, NULL, NULL);

    for (int i = 0; i < 10; ++i) {
        printf("%0.2f + %0.2f = %0.2f\n", host_a[i], host_b[i], host_c[i]);
    }

    // 14) Libération des espaces mémoires des deux côtés
    clReleaseEvent(kernel_event);
    clReleaseMemObject(a);
    clReleaseMemObject(b);
    clReleaseMemObject(c);
    free(host_a);
    free(host_b);
    free(host_c);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
    return 0;
}
