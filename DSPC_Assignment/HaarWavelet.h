#pragma once
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <mpi.h>
#include <cuda.h>
#include <CL/cl.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define DELTA 1
#define DIM_MASCARA 8
#define LONG_MASCARA 64
#define MAX_SYMBOLS 256 // Adjust as needed
#define TARGET_SIZE 16

namespace HaarWaveletWrapper
{
	void MultIzquierdaGPU(dim3 gridSize, dim3 blockSize, float* const imagenOriginal, float* const imagenSalida,
		float* const mascara, int filas, int columnas, bool setQuantization);

	void MultDerechaGPU(dim3 gridSize, dim3 blockSize, float* const imagenOriginal, float* const imagenSalida,
		float* const mascara, int filas, int columnas, bool setQuantization);
};