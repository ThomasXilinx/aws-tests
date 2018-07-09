
#include <stdio.h>
#include "ap_int.h"
#include "hls_stream.h"

const unsigned int MAX_LATENCY = 1024;
typedef ap_uint<512> uint512_dt;

extern "C" {
void CopyKernel(
        const uint512_dt *src,
        uint512_dt       *dst,      
        unsigned int      nvalues,
        unsigned int      latency               
        )
{
    #pragma HLS INTERFACE m_axi     port=src offset=slave bundle=gmem0 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi     port=dst offset=slave bundle=gmem1 max_write_burst_length=256
    #pragma HLS INTERFACE s_axilite port=src              bundle=control
    #pragma HLS INTERFACE s_axilite port=dst              bundle=control
    #pragma HLS INTERFACE s_axilite port=nvalues          bundle=control
    #pragma HLS INTERFACE s_axilite port=latency          bundle=control
    #pragma HLS INTERFACE s_axilite port=return           bundle=control

    hls::stream<uint512_dt> rdata;
    hls::stream<uint512_dt> wdata;
    hls::stream<uint512_dt> delay;
    #pragma HLS stream variable=rdata depth=8
    #pragma HLS stream variable=wdata depth=8
    #pragma HLS stream variable=delay depth=MAX_LATENCY

    latency = (latency > MAX_LATENCY) ? MAX_LATENCY : latency;

    #pragma HLS DATAFLOW

    for (int i=0; i<nvalues; i++) 
    {
        #pragma HLS PIPELINE II=1
        rdata.write( src[i] );
    }

    for (int i=0; i<nvalues+latency; i++) 
    {
        #pragma HLS PIPELINE II=1
        if (i<nvalues) {
            delay.write(rdata.read());
        } 
        if (i>=latency) {
            wdata.write(delay.read());
        }
    }

    for (int i=0; i<nvalues; i++) 
    {
        #pragma HLS PIPELINE II=1
        dst[i] = wdata.read();
    }    
}
}
