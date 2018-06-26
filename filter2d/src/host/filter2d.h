#pragma once

#include <algorithm>

using namespace std;

#define FILTER2D_KERNEL_H_SIZE 	15
#define FILTER2D_KERNEL_V_SIZE 	15

//#define MIN(a,b) ((a<b)?a:b)
//#define MAX(a,b) ((a<b)?b:a)


void Filter2D(
        const    short coeffs[FILTER2D_KERNEL_V_SIZE][FILTER2D_KERNEL_H_SIZE],
		float		   factor,
		unsigned char *srcImg,
		unsigned int   width,
		unsigned int   height,
		unsigned int   stride,
		unsigned char *dstImg );

