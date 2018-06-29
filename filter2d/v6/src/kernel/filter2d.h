#pragma once

#define FILTER_KERNEL_V_SIZE	15
#define FILTER_KERNEL_H_SIZE	15

#define MIN(a,b) ((a<b)?a:b)
#define MAX(a,b) ((a<b)?b:a)

#include <stdio.h>
#include <string.h>
#include "axi2stream.h"

extern "C" {

void Filter2DKernel(
        const short* coeffs,
        float        factor,
		const ap_uint<AXIMM_DATA_WIDTH>* src,
		unsigned int width,
		unsigned int height,
		unsigned int stride,
		ap_uint<AXIMM_DATA_WIDTH>* dst );
}
