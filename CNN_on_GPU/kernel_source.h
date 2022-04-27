#pragma once

const char* KernelSource = "\n" \
"//maxPool2d                                                                                                \n"\
"//kernel_size=3 stride=2                                                                                   \n"\
"//output one feature map per kernel                                                                        \n"\
"__kernel void maxpool2d(                                                                                   \n"\
"	const int input_size,                                                                                   \n"\
"	const int output_size,                                                                                  \n"\
"	__global float *input_im,                                                                               \n"\
"    __global float *restrict output_im)                                                                    \n"\
"{                                                                                                          \n"\
"	int channels = get_global_id(0);//get output channel index                                              \n"\
"	                                                                                                        \n"\
"	input_im += channels * input_size * input_size;                                                         \n"\
"	output_im += channels * output_size * output_size;                                                      \n"\
"                                                                                                           \n"\
"	//loop over output feature map                                                                          \n"\
"	for(int i = 0; i < output_size; i++)//row                                                               \n"\
"	{                                                                                                       \n"\
"		for(int j = 0; j < output_size; j++)//col                                                           \n"\
"		{                                                                                                   \n"\
"			//find the max value in 3x3 reigon                                                              \n"\
"			//to be one element in the output feature map                                                   \n"\
"			float tmp = 0.0;                                                                                \n"\
"                                                                                                           \n"\
"			#pragma unroll 1                                                                                \n"\
"			for(int k = 0; k < 3; k++)//row                                                                 \n"\
"			{                                                                                               \n"\
"				#pragma unroll 1                                                                            \n"\
"				for(int l = 0; l < 3; l++)//col                                                             \n"\
"				{                                                                                           \n"\
"					float value = input_im[(i * 2 + k) * input_size  + j * 2 + l ];                         \n"\
"					if(value > tmp)                                                                         \n"\
"						tmp = value;                                                                        \n"\
"				}                                                                                           \n"\
"			}                                                                                               \n"\
"			//store the result to output feature map                                                        \n"\
"			output_im[i * output_size + j] = tmp;                                                           \n"\
"		}                                                                                                   \n"\
"	}                                                                                                       \n"\
"}                                                                                                          \n"\
"                                                                                                           \n"\
"//3x3 convolution layer                                                                                    \n"\
"//output one feature map per kernel                                                                        \n"\
"__kernel void conv2d3x3(                                                                                   \n"\
"	const int input_channels, const int input_size,                                                         \n"\
"	const int pad, const int stride,                                                                        \n"\
"	const int start_channel, //start_channel is for 1x1 feature map in fire layer                           \n"\
"	const int output_size,                                                                                  \n"\
"	__global float* input_im,                                                                               \n"\
"	__global const float* filter_weight,                                                                    \n"\
"	__global const float* filter_bias,                                                                      \n"\
"	__global float *restrict output_im                                                                      \n"\
"	)                                                                                                       \n"\
"{                                                                                                          \n"\
"	int filter_index = get_global_id(0); //get output channel index                                         \n"\
"	int i =  get_global_id(1);                                                                              \n"\
"                                                                                                           \n"\
"	filter_weight += filter_index * input_channels * 9;                                                     \n"\
"	float bias = filter_bias[filter_index];                                                                 \n"\
"	output_im += (start_channel + filter_index) * output_size * output_size;                                \n"\
"	                                                                                                        \n"\
"	//loop over output feature map                                                                          \n"\
"	//for(int i = 0; i < output_size; i++)                                                                  \n"\
"	{                                                                                                       \n"\
"		for(int j = 0; j < output_size; j++)                                                                \n"\
"		{                                                                                                   \n"\
"			//compute one element in the output feature map                                                 \n"\
"			float tmp = bias;                                                                               \n"\
"			                                                                                                \n"\
"			//compute dot product of 2 input_channels x 3 x 3 matrix                                        \n"\
"			for(int k = 0; k < input_channels; k++)                                                         \n"\
"			{                                                                                               \n"\
"				#pragma unroll                                                                              \n"\
"				for(int l = 0; l < 3; l++)                                                                  \n"\
"				{                                                                                           \n"\
"					int h = i * stride + l - pad;                                                           \n"\
"					for(int m = 0; m < 3; m++)                                                              \n"\
"					{                                                                                       \n"\
"						int w = j * stride + m - pad;                                                       \n"\
"						if((h >= 0) && (h < input_size) && (w >= 0) && (w < input_size))                    \n"\
"						{                                                                                   \n"\
"							tmp += input_im[k * input_size * input_size + h * input_size + w] \             \n"\
"                               * filter_weight[9 * k + 3 * l + m];                                         \n"\
"						}                                                                                   \n"\
"					}                                                                                       \n"\
"				}                                                                                           \n"\
"			}                                                                                               \n"\
"                                                                                                           \n"\
"			//add relu activation after conv                                                                \n"\
"			output_im[i * output_size + j] = (tmp > 0.0) ? tmp : 0.0;                                       \n"\
"		}                                                                                                   \n"\
"	}                                                                                                       \n"\
"}                                                                                                          \n"\
"                                                                                                           \n"\
"//1x1 convolution layer                                                                                    \n"\
"//output one feature map per kernel                                                                        \n"\
"__kernel void conv2d1x1(                                                                                   \n"\
"	const int input_channels, const int input_size,                                                         \n"\
"	__global float *input_im,                                                                               \n"\
"	__global const float4* filter_weight,                                                                   \n"\
"	__global const float* filter_bias,                                                                      \n"\
"	__global float *restrict output_im)                                                                     \n"\
"{                                                                                                          \n"\
"	int filter_index = get_global_id(0); // 0 - (output_channels - 1)                                       \n"\
"	int i = get_global_id(1);                                                                               \n"\
"                                                                                                           \n"\
"	filter_weight += filter_index * input_channels;                                                         \n"\
"                                                                                                           \n"\
"	float bias = filter_bias[filter_index];                                                                 \n"\
"	                                                                                                        \n"\
"	output_im += filter_index * input_size * input_size;//start_channel is for 1x1 feature map in fire layer\n"\
"                                                                                                           \n"\
"	//loop over output feature map                                                                          \n"\
"	//for(int i = 0; i < input_size; i++)                                                                   \n"\
"	{                                                                                                       \n"\
"		for(int j = 0; j < input_size; j++)                                                                 \n"\
"		{                                                                                                   \n"\
"			float tmp = bias;                                                                               \n"\
"			int loc = i * input_size + j;                                                                   \n"\
"                                                                                                           \n"\
"			for(int k = 0; k < input_channels; k++)                                                         \n"\
"			{                                                                                               \n"\
"				//float8 weight = filter_weight[k];                                                         \n"\
"				//float8 feature;                                                                           \n"\
"				tmp += input_im[((k << 2) + 0) * input_size * input_size + loc] * filter_weight[k].s0       \n"\
"				     + input_im[((k << 2) + 1) * input_size * input_size + loc] * filter_weight[k].s1       \n"\
"					 + input_im[((k << 2) + 2) * input_size * input_size + loc] * filter_weight[k].s2       \n"\
"					 + input_im[((k << 2) + 3) * input_size * input_size + loc] * filter_weight[k].s3;      \n"\
"			}                                                                                               \n"\
"			//add relu after conv                                                                           \n"\
"			output_im[i * input_size + j] = (tmp > 0.0) ? tmp : 0.0;                                        \n"\
"		}                                                                                                   \n"\
"	}                                                                                                       \n"\
"}                                                                                                          \n"\
"                                                                                                           \n"\
"//last layer use a 13 x 13 avgPool layer as classifier                                                     \n"\
"//one class score per kernel                                                                               \n"\
"__kernel void avgpool2d(                                                                                   \n"\
"	__global float* input_im,                                                                               \n"\
"	__global float *restrict output_im)                                                                     \n"\
"{                                                                                                          \n"\
"	int class_index = get_global_id(0);//get class score index                                              \n"\
"                                                                                                           \n"\
"	input_im += 169 * class_index;                                                                          \n"\
"	                                                                                                        \n"\
"	float tmp = 0.0f;                                                                                       \n"\
"                                                                                                           \n"\
"	for(int i = 0; i < 169; i++)                                                                            \n"\
"	{                                                                                                       \n"\
"		tmp += input_im[i];                                                                                 \n"\
"	}                                                                                                       \n"\
"                                                                                                           \n"\
"	output_im[class_index] = tmp / 169.0;                                                                   \n"\
"}                                                                                                          \n"\
"\n";