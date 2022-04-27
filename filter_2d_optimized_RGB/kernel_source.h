const char* KernelSource3 = "\n" \
"__kernel void __attribute__ ((reqd_work_group_size(32,32,1))) convolute(                                                                                                            \n"\
"	const __global float * input,                                                                                                   \n"\
"	__constant float * filter,                                                                                                      \n"\
"	__global float * output,                                                                                                        \n"\
"	__local float * cached                                                                                                          \n"\
")                                                                                                                                   \n"\
"{                                                                                                                                   \n"\
"	const int rowOffset = get_global_id(1) * IMAGE_W;                                                                               \n"\
"	const int my = get_global_id(0) + rowOffset;                                                                                    \n"\
"	                                                                                                                                \n"\
"	const int localRowLen = TWICE_HALF_FILTER_SIZE + get_local_size(0);                                                             \n"\
"	const int localRowOffset = ( get_local_id(1) + HALF_FILTER_SIZE ) * localRowLen;                                                \n"\
"	const int myLocal = localRowOffset + get_local_id(0) + HALF_FILTER_SIZE;		                                                \n"\
"		                                                                                                                            \n"\
"	// copy my pixel                                                                                                                \n"\
"	cached[ myLocal ] = input[ my ];                                                                                                \n"\
"                                                                                                                                    \n"\
"	                                                                                                                                \n"\
"	if (                                                                                                                            \n"\
"		get_global_id(0) < HALF_FILTER_SIZE 			||                                                                          \n"\
"		get_global_id(0) > IMAGE_W - HALF_FILTER_SIZE - 1		||                                                                  \n"\
"		get_global_id(1) < HALF_FILTER_SIZE			||                                                                              \n"\
"		get_global_id(1) > IMAGE_H - HALF_FILTER_SIZE - 1                                                                           \n"\
"	)                                                                                                                               \n"\
"	{                                                                                                                               \n"\
"		// no computation for me, sync and exit                                                                                     \n"\
"		barrier(CLK_LOCAL_MEM_FENCE);                                                                                               \n"\
"		return;                                                                                                                     \n"\
"	}                                                                                                                               \n"\
"	else                                                                                                                            \n"\
"	{                                                                                                                               \n"\
"		// copy additional elements                                                                                                 \n"\
"		int localColOffset = -1;                                                                                                    \n"\
"		int globalColOffset = -1;                                                                                                   \n"\
"		                                                                                                                            \n"\
"		if ( get_local_id(0) < HALF_FILTER_SIZE )                                                                                   \n"\
"		{                                                                                                                           \n"\
"			localColOffset = get_local_id(0);                                                                                       \n"\
"			globalColOffset = -HALF_FILTER_SIZE;                                                                                    \n"\
"			                                                                                                                        \n"\
"			cached[ localRowOffset + get_local_id(0) ] = input[ my - HALF_FILTER_SIZE ];                                            \n"\
"		}                                                                                                                           \n"\
"		else if ( get_local_id(0) >= get_local_size(0) - HALF_FILTER_SIZE )                                                         \n"\
"		{                                                                                                                           \n"\
"			localColOffset = get_local_id(0) + TWICE_HALF_FILTER_SIZE;                                                              \n"\
"			globalColOffset = HALF_FILTER_SIZE;                                                                                     \n"\
"			                                                                                                                        \n"\
"			cached[ myLocal + HALF_FILTER_SIZE ] = input[ my + HALF_FILTER_SIZE ];                                                  \n"\
"		}                                                                                                                           \n"\
"		                                                                                                                            \n"\
"		                                                                                                                            \n"\
"		if ( get_local_id(1) < HALF_FILTER_SIZE )                                                                                   \n"\
"		{                                                                                                                           \n"\
"			cached[ get_local_id(1) * localRowLen + get_local_id(0) + HALF_FILTER_SIZE ] = input[ my - HALF_FILTER_SIZE_IMAGE_W ];  \n"\
"			if (localColOffset > 0)                                                                                                 \n"\
"			{                                                                                                                       \n"\
"				cached[ get_local_id(1) * localRowLen + localColOffset ] = input[ my - HALF_FILTER_SIZE_IMAGE_W + globalColOffset ];\n"\
"			}                                                                                                                       \n"\
"		}                                                                                                                           \n"\
"		else if ( get_local_id(1) >= get_local_size(1) -HALF_FILTER_SIZE )                                                          \n"\
"		{                                                                                                                           \n"\
"			int offset = ( get_local_id(1) + TWICE_HALF_FILTER_SIZE ) * localRowLen;                                                \n"\
"			cached[ offset + get_local_id(0) + HALF_FILTER_SIZE ] = input[ my + HALF_FILTER_SIZE_IMAGE_W ];                         \n"\
"			if (localColOffset > 0)                                                                                                 \n"\
"			{                                                                                                                       \n"\
"				cached[ offset + localColOffset ] = input[ my + HALF_FILTER_SIZE_IMAGE_W + globalColOffset ];                       \n"\
"			}                                                                                                                       \n"\
"		}                                                                                                                           \n"\
"		                                                                                                                            \n"\
"		// sync                                                                                                                     \n"\
"		barrier(CLK_LOCAL_MEM_FENCE);                                                                                               \n"\
"                                                                                                                                    \n"\
"		                                                                                                                            \n"\
"		// perform convolution                                                                                                      \n"\
"		int fIndex = 0;                                                                                                             \n"\
"		float sum = (float) 0.0;                                                                                                    \n"\
"		                                                                                                                            \n"\
"		for (int r = -HALF_FILTER_SIZE; r <= HALF_FILTER_SIZE; r++)                                                                 \n"\
"		{                                                                                                                           \n"\
"			int curRow = r * localRowLen;                                                                                           \n"\
"			for (int c = -HALF_FILTER_SIZE; c <= HALF_FILTER_SIZE; c++, fIndex++)                                                   \n"\
"			{	                                                                                                                    \n"\
"				sum += cached[ myLocal + curRow + c ] * filter[ fIndex ];                                                           \n"\
"			}                                                                                                                       \n"\
"		}                                                                                                                           \n"\
"		output[my] = sum;                                                                                                           \n"\
"	}                                                                                                                               \n"\
"}                                                                                                                                   \n"\
"\n";


const char* KernelSource2 = "\n" \
"__kernel void __attribute__ ((reqd_work_group_size(32,32,1))) convolute(                                                                     \n"\
"   const __global float* im_i,                                                 \n" \
"   __constant float* filter_i,                                                             \n" \
"   __global float* im_o,                                                                   \n" \
"   __local float * cached)                                                                  \n" \
"{                                                                                           \n" \
"   int x = get_global_id(0);                                           \n" \
"   int y = get_global_id(1);                                           \n" \
"   float sum = 0.0;														\n" \
"   //#pragma unroll													\n" \
"   for(int j = 0 ; j < FILTER_SIZE ; j++){                                       \n" \
"		//#pragma unroll														\n" \
"		for(int i = 0 ; i < FILTER_SIZE ; i++){                                   \n" \
"			sum += filter_i[i + j*FILTER_SIZE] * im_i[x + y*IMAGE_W + i + j*IMAGE_W];         \n" \
"		}                                                               \n" \
"   }                                                                   \n" \
"   im_o[x + y * IMAGE_W] = sum;										\n" \
"}                                                                      \n" \
"\n";

const char* KernelSource = "\n" \
"__kernel void /*__attribute__ ((reqd_work_group_size(32,32,1)))*/ convolute(                                                              \n"\
"	const __global float *input,                                                                                                  \n"\
"	__constant float *filter,                                                                                                     \n"\
"	__global float *output,                                                                                                       \n"\
"	__local float *cached                                                                                                         \n"\
")                                                                                                                                 \n"\
"{                                                                                                                                 \n"\
"	const int rowOffset = get_global_id(1) * IMAGE_W;                                                                             \n"\
"	const int my = get_global_id(0) + rowOffset;                                                                                  \n"\
"	                                                                                                                              \n"\
"	const int localRowLen = HALF_FILTER_SIZE * 2 + get_local_size(0);                                                             \n"\
"	const int localRowOffset = ( get_local_id(1) + HALF_FILTER_SIZE ) * localRowLen;                                              \n"\
"	const int myLocal = localRowOffset + get_local_id(0) + HALF_FILTER_SIZE;		                                              \n"\
"	// copy my pixel                                                                                                              \n"\
"	cached[ myLocal ] = input[ my ];                                                                                              \n"\
"                                                                                                                                  \n"\
"    // copy additional elements                                                                                                   \n"\
"    int localColOffset = -1;                                                                                                      \n"\
"    int globalColOffset = -1;                                                                                                     \n"\
"                                                                                                                                  \n"\
"    if ( get_local_id(0) < HALF_FILTER_SIZE )                                                                                     \n"\
"    {                                                                                                                             \n"\
"        localColOffset = get_local_id(0);                                                                                         \n"\
"        globalColOffset = -HALF_FILTER_SIZE;                                                                                      \n"\
"                                                                                                                                  \n"\
"        cached[ localRowOffset + get_local_id(0)] = input[ my - HALF_FILTER_SIZE ];                                              \n"\
"    }                                                                                                                             \n"\
"    else if ( get_local_id(0) >= get_local_size(0) -HALF_FILTER_SIZE )                                                                 \n"\
"    {                                                                                                                             \n"\
"        localColOffset = get_local_id(0) + HALF_FILTER_SIZE * 2;                                                                  \n"\
"        globalColOffset = HALF_FILTER_SIZE;                                                                                       \n"\
"                                                                                                                                  \n"\
"        cached[ localRowOffset + get_local_id(0) + HALF_FILTER_SIZE * 2 ] = input[ my + HALF_FILTER_SIZE ];                       \n"\
"    }                                                                                                                             \n"\
"                                                                                                                                  \n"\
"                                                                                                                                  \n"\
"    if ( get_local_id(1) < HALF_FILTER_SIZE )                                                                                     \n"\
"    {                                                                                                                             \n"\
"        cached[ get_local_id(1) * localRowLen + get_local_id(0) + HALF_FILTER_SIZE ] = input[ my - HALF_FILTER_SIZE * IMAGE_W ];  \n"\
"        if (localColOffset >= 0)                                                                                                   \n"\
"        {                                                                                                                         \n"\
"            cached[ get_local_id(1) * localRowLen + localColOffset ] = input[ my - HALF_FILTER_SIZE * IMAGE_W + globalColOffset ];\n"\
"        }                                                                                                                         \n"\
"    }                                                                                                                             \n"\
"    else if ( get_local_id(1) >= get_local_size(1) -HALF_FILTER_SIZE )                                                                 \n"\
"    {                                                                                                                             \n"\
"        int offset = ( get_local_id(1) + 2 * HALF_FILTER_SIZE ) * localRowLen;                                                    \n"\
"        cached[ offset + get_local_id(0) + HALF_FILTER_SIZE ] = input[ my + HALF_FILTER_SIZE * IMAGE_W ];                         \n"\
"        if (localColOffset >= 0)                                                                                                   \n"\
"        {                                                                                                                         \n"\
"            cached[ offset + localColOffset ] = input[ my + HALF_FILTER_SIZE * IMAGE_W + globalColOffset ];                       \n"\
"        }                                                                                                                         \n"\
"    }                                                                                                                             \n"\
"                                                                                                                                  \n"\
"    // sync                                                                                                                       \n"\
"    barrier(CLK_LOCAL_MEM_FENCE);                                                                                                 \n"\
"                                                                                                                                  \n"\
"	if (                                                                                                                          \n"\
"		get_global_id(0) < HALF_FILTER_SIZE ||                                                                                    \n"\
"		get_global_id(0) > IMAGE_W - HALF_FILTER_SIZE -1||                                                                        \n"\
"		get_global_id(1) < HALF_FILTER_SIZE ||                                                                                    \n"\
"		get_global_id(1) > IMAGE_H - HALF_FILTER_SIZE -1                                                                          \n"\
"	)                                                                                                                             \n"\
"	{                                                                                                                             \n"\
"		// no computation for me, sync and exit                                                                                   \n"\
"		return;                                                                                                                   \n"\
"	}                                                                                                                             \n"\
"	else                                                                                                                          \n"\
"    {                                                                                                                             \n"\
"		                                                                                                                          \n"\
"        // perform convolution                                                                                                    \n"\
"		int fIndex = 0;                                                                                                           \n"\
"		float sum = (float) 0.0;                                                                                                  \n"\
"		                                                                                                                          \n"\
"		for (int r = -HALF_FILTER_SIZE; r <= HALF_FILTER_SIZE; r++)                                                               \n"\
"		{                                                                                                                         \n"\
"			int curRow = r * localRowLen;                                                                                         \n"\
"			for (int c = -HALF_FILTER_SIZE; c <= HALF_FILTER_SIZE; c++, fIndex++)                                                 \n"\
"			{	                                                                                                                  \n"\
"				sum += cached[ myLocal + curRow + c ] * filter[ fIndex ];                                                         \n"\
"			}                                                                                                                     \n"\
"		}                                                                                                                         \n"\
"		output[my] = sum;                                                                                                         \n"\
"	}                                                                                                                             \n"\
"}                                                                                                                                \n"\
"\n";