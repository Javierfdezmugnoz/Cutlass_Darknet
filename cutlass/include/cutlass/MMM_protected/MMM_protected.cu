// Standard Library includes
#pragma once
#include <iostream>
#include <sstream>
#include <vector>

#include <sched.h>
#include <string.h>

// Defines cutlass::gemm::device::Gemm, the generic Gemm computation template class.
#include "MMM_protected.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// This function defines a CUTLASS GEMM kernel instantiation, constructs its parameters object,
// and launches it on the CUDA device.
//
///////////////////////////////////////////////////////////////////////////////////////////////////
#include "cuda_runtime.h"


/* ==========================================================================
  Descritption: Addition of two values using PTX (parallel thread execution) 
  and ISA (parallel thread execution with instruction set architecture) adding
  the carry bit. After that, one's complement is implemented (bit negation). 
  Additional info:
  url: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html or
  https://docs.nvidia.com/pdf/ptx_isa_5.0.pdf 
=============================================================================*/
__forceinline__ __device__ uint32_t _xor (uint32_t ui32_a, uint32_t ui32_b)
{
    uint32_t acc;
    asm ("xor.b32  %0, %1, %2;\n\t"
                : "=r"(acc)
                : "r"(ui32_a), "r"(ui32_b));
    return acc;
}


/* ==========================================================================
  Descritption: Addition of two values using PTX (parallel thread execution) 
  and ISA (parallel thread execution with instruction set architecture) adding
  the carry bit. After that, one's complement is implemented (bit negation). 
  Additional info:
  url: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html or
  https://docs.nvidia.com/pdf/ptx_isa_5.0.pdf 
=============================================================================*/
__forceinline__ __device__ uint32_t __a1c (uint32_t ui32_a, uint32_t ui32_b)
{
    uint32_t acc;
    asm ("add.cc.u32      %0, %1, %2;\n\t"
         "addc.u32        %0, %0, 0;\n\t"
         "not.b32         %0, %0;\n\t"
         : "=r"(acc)
         : "r"(ui32_a), "r"(ui32_b));
    return acc;
}


/* ==========================================================================
  Descritption: Ones complement with atomic instructions
=============================================================================*/
__forceinline__ __device__ uint32_t a1c_atomic (uint32_t ui32_a, uint32_t ui32_b)
{
    atomicAdd((uint32_t*) &ui32_a, ui32_b);
    ui32_a = ~ui32_a;
    return ui32_a;
}


/* ==========================================================================
  Description: Addition of two values using PTX (parallel thread execution) 
  and ISA (parallel thread execution with instruction set architecture) with
  not carry-bit addition. This operation is followed by the two's complement
  implementation (bit negation and then, addition of 1)
  Additional info:
  url: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html or
  https://docs.nvidia.com/pdf/ptx_isa_5.0.pdf
=============================================================================*/
__forceinline__ __device__ uint32_t __a2c (uint32_t ui32_a, uint32_t ui32_b)
{
    uint32_t acc = 0;
    asm ("add.u32     %0, %1, %2;\n\t"
         "not.b32     %0, %0;\n\t"
         "add.u32     %0, %0, 1;\n\t"
         : "=r"(acc)
         : "r"(ui32_a), "r"(ui32_b));
    return acc;
}

/* ==========================================================================
  Descritption: Two's complement with atomic instructions
=============================================================================*/
__forceinline__ __device__ uint32_t a2c_atomic (uint32_t ui32_a, uint32_t ui32_b)
{
    atomicAdd((uint32_t*) &ui32_a, ui32_b);
    ui32_a = ~ui32_a;
    atomicAdd((uint32_t*) &ui32_a, 1u);
    return ui32_a;
}

/* ==========================================================================
  Description: CRC checksum
=============================================================================*/
#if ((INTERNAL_ES==CRC_CHECKSUM) || (INTERMEDIATE_ES==CRC_CHECKSUM) || (CRC_CHECKSUM==EXTERNAL_ES))
__forceinline__ __device__  uint32_t singletable_crc32c_ui32(uint32_t *d_CRC_table_shared, uint32_t ui32_crc, uint32_t ui32_data) //
  {
    ui32_to_ui8_t u;
    u.ui32 = ui32_data;

    /* 4 bytes*/
    ui32_crc = d_CRC_table_shared[(ui32_crc ^ u.ui8[0u]) & 0x00ffu] ^ (ui32_crc >> 8u);
    ui32_crc = d_CRC_table_shared[(ui32_crc ^ u.ui8[1u]) & 0x00ffu] ^ (ui32_crc >> 8u);
    ui32_crc = d_CRC_table_shared[(ui32_crc ^ u.ui8[2u]) & 0x00ffu] ^ (ui32_crc >> 8u);
    ui32_crc = d_CRC_table_shared[(ui32_crc ^ u.ui8[3u]) & 0x00ffu] ^ (ui32_crc >> 8u);
    return ui32_crc;
  }
#endif

/* ==========================================================================
  Description: Fletcher checksum
=============================================================================*/
__forceinline__ __device__ uint32_t Fletcher32c_ui32(uint32_t Prev_Fletcher, uint32_t ui32_data)
{
	  ui32_to_ui16_t v;
    ui32_to_ui16_t Fletcher;
    v.ui32        = ui32_data;
    Fletcher.ui32 = Prev_Fletcher;

	Fletcher.ui16[0] += v.ui16[0];
	Fletcher.ui16[1] += Fletcher.ui16[0];
	Fletcher.ui16[0] += v.ui16[1];
	Fletcher.ui16[1] += Fletcher.ui16[0];
  /*
  Fletcher.ui16[0] = Fletcher.ui16[0] % 255
  Fletcher.ui16[0] = Fletcher.ui16[0] & (FLET_DIVISOR-1)
  */

	Fletcher.ui16[0] %= 255;
	Fletcher.ui16[1] %= 255;

	return Fletcher.ui32;
}


// ==========================================================================
//  Descritption: Two's complement with atomic instructions
// ==========================================================================
 __forceinline__ __device__ uint32_t get_smid(void) 
{
     uint32_t ret;
     asm("mov.u32 %0, %%smid;" : "=r"(ret) );
     return ret;
}