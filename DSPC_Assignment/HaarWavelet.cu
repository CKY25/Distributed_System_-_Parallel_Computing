#include "HaarWavelet.h"

__global__ void multIzquierdaGPU(float* const imagenOriginal, float* const imagenSalida,
	float* const mascara, int filas, int columnas, bool setQuantization) {
	int fila = (blockIdx.x * blockDim.x) + threadIdx.x;
	int columna = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (columna >= columnas || fila >= filas) {
		return;
	}

	int indice = fila * columnas + columna;

	int filaIni = (blockIdx.x * blockDim.x);
	int columIni = (blockIdx.y * blockDim.y);

	int r = threadIdx.x;  /// renglon inicial del bloque 8 x 8
	int c = threadIdx.y;  /// columna inicial del bloque 8 x 8

	float total = 0;
	for (int k = 0; k < DIM_MASCARA; k++) {
		int indMascara = (k * DIM_MASCARA) + c;
		int indImagen = ((filaIni + r) * filas) + (k + columIni);
		total = total + mascara[indMascara] * imagenOriginal[indImagen];
	}
	if (setQuantization)
		imagenSalida[indice] = round(total / DELTA);
	else
		imagenSalida[indice] = total;
}

void HaarWaveletWrapper::MultIzquierdaGPU(dim3 gridSize, dim3 blockSize, float* const imagenOriginal, float* const imagenSalida, float* const mascara, int filas, int columnas, bool setQuantization)
{
	multIzquierdaGPU <<<gridSize, blockSize>>>
		(imagenOriginal, imagenSalida, mascara,
			filas, columnas, setQuantization);
}

__global__ void multDerechaGPU(float* const imagenOriginal, float* const imagenSalida,
	float* const mascara, int filas, int columnas, bool setQuantization) {
	int fila = (blockIdx.x * blockDim.x) + threadIdx.x;
	int columna = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (columna >= columnas || fila >= filas) {
		return;
	}

	int indice = fila * columnas + columna;

	int filaIni = (blockIdx.x * blockDim.x);
	int columIni = (blockIdx.y * blockDim.y);

	int r = threadIdx.x;  /// renglon inicial del bloque 8 x 8
	int c = threadIdx.y;  /// columna inicial del bloque 8 x 8

	float total = 0;
	for (int k = 0; k < DIM_MASCARA; k++) {
		int indMascara = (r * DIM_MASCARA) + k;
		int indImagen = ((filaIni + k) * columnas) + (c + columIni);
		if (setQuantization)
			total = total + mascara[indMascara] * imagenOriginal[indImagen];
		else
			total = total + mascara[indMascara] * (imagenOriginal[indImagen] * DELTA);
	}
	imagenSalida[indice] = total;
}

void HaarWaveletWrapper::MultDerechaGPU(dim3 gridSize, dim3 blockSize, float* const imagenOriginal, float* const imagenSalida, float* const mascara, int filas, int columnas, bool setQuantization)
{
	multDerechaGPU <<<gridSize, blockSize>>>
		(imagenOriginal, imagenSalida, mascara,
			filas, columnas, setQuantization);
}