#include <filesystem>
#include <fstream>
#include <queue>
#include <unordered_map>
#include <vector>
#include "HaarWavelet.h"

using namespace cv;
using namespace std;

// quality-metric
namespace qm {
#define C1 (float) (0.01 * 255 * 0.01  * 255)
#define C2 (float) (0.03 * 255 * 0.03  * 255)

	// sigma on block_size
	double sigma(Mat m, int i, int j, int block_size)
	{
		double sd = 0;

		Mat m_tmp = m(Range(i, i + block_size), Range(j, j + block_size));
		Mat m_squared(block_size, block_size, CV_64F);

		multiply(m_tmp, m_tmp, m_squared);

		// E(x)
		double avg = mean(m_tmp)[0];
		// E(x²)
		double avg_2 = mean(m_squared)[0];

		sd = sqrt(avg_2 - avg * avg);

		return sd;
	}

	// Covariance
	double cov(Mat m1, Mat m2, int i, int j, int block_size)
	{
		Mat m3 = Mat::zeros(block_size, block_size, m1.depth());
		Mat m1_tmp = m1(Range(i, i + block_size), Range(j, j + block_size));
		Mat m2_tmp = m2(Range(i, i + block_size), Range(j, j + block_size));

		multiply(m1_tmp, m2_tmp, m3);

		double avg_ro = mean(m3)[0]; // E(XY)
		double avg_r = mean(m1_tmp)[0]; // E(X)
		double avg_o = mean(m2_tmp)[0]; // E(Y)

		double sd_ro = avg_ro - avg_o * avg_r; // E(XY) - E(X)E(Y)

		return sd_ro;
	}

	// Mean squared error
	double eqm(Mat img1, Mat img2)
	{
		int i, j;
		double eqm = 0;
		int height = img1.rows;
		int width = img1.cols;

		for (i = 0; i < height; i++)
			for (j = 0; j < width; j++)
				eqm += (img1.at<double>(i, j) - img2.at<double>(i, j)) * (img1.at<double>(i, j) - img2.at<double>(i, j));

		eqm /= height * width;

		return eqm;
	}

	/**
	 *    Compute the PSNR between 2 images
	 */
	double psnr(Mat img_src, Mat img_compressed, int block_size)
	{
		int D = 255;
		return (10 * log10((D * D) / eqm(img_src, img_compressed)));
	}

	/**
	 * Compute the SSIM between 2 images
	 */
	double ssim(Mat img_src, Mat img_compressed, int block_size, bool show_progress = false)
	{
		double ssim = 0;

		int nbBlockPerHeight = img_src.rows / block_size;
		int nbBlockPerWidth = img_src.cols / block_size;

		for (int k = 0; k < nbBlockPerHeight; k++)
		{
			for (int l = 0; l < nbBlockPerWidth; l++)
			{
				int m = k * block_size;
				int n = l * block_size;

				double avg_o = mean(img_src(Range(k, k + block_size), Range(l, l + block_size)))[0];
				double avg_r = mean(img_compressed(Range(k, k + block_size), Range(l, l + block_size)))[0];
				double sigma_o = sigma(img_src, m, n, block_size);
				double sigma_r = sigma(img_compressed, m, n, block_size);
				double sigma_ro = cov(img_src, img_compressed, m, n, block_size);

				ssim += ((2 * avg_o * avg_r + C1) * (2 * sigma_ro + C2)) / ((avg_o * avg_o + avg_r * avg_r + C1) * (sigma_o * sigma_o + sigma_r * sigma_r + C2));
			}
			// Progress
			if (show_progress)
				cout << "\r>>SSIM [" << (int)((((double)k) / nbBlockPerHeight) * 100) << "%]";
		}
		ssim /= nbBlockPerHeight * nbBlockPerWidth;

		if (show_progress)
		{
			cout << "\r>>SSIM [100%]" << endl;
			cout << "SSIM : " << ssim << endl;
		}

		return ssim;
	}
}

Mat toZigZag(Mat imageMatrix) {
	Mat imageArray = Mat::zeros(1, imageMatrix.cols * imageMatrix.rows, CV_32F);

	int indexImageArray = 0;

	int index_X = 0;
	int index_Y = 0;

	while (indexImageArray < (imageMatrix.cols * imageMatrix.rows)) {
		if (index_X < imageMatrix.cols - 1) {
			imageArray.at<float>(0, indexImageArray) = imageMatrix.at<float>(index_X, index_Y);
			indexImageArray++;
			index_X++;

			while (index_X > 0) {
				imageArray.at<float>(0, indexImageArray) = imageMatrix.at<float>(index_X, index_Y);
				indexImageArray++;
				index_X--;
				index_Y++;
			}
		}
		else if (index_X == imageMatrix.cols - 1) {
			imageArray.at<float>(0, indexImageArray) = imageMatrix.at<float>(index_X, index_Y);
			indexImageArray++;
			index_Y++;

			while (index_Y < imageMatrix.rows - 1) {
				imageArray.at<float>(0, indexImageArray) = imageMatrix.at<float>(index_X, index_Y);
				indexImageArray++;
				index_X--;
				index_Y++;
			}
		}
		if (index_Y < imageMatrix.rows - 1) {
			imageArray.at<float>(0, indexImageArray) = imageMatrix.at<float>(index_X, index_Y);
			indexImageArray++;
			index_Y++;

			while (index_Y > 0) {
				imageArray.at<float>(0, indexImageArray) = imageMatrix.at<float>(index_X, index_Y);
				indexImageArray++;
				index_Y--;
				index_X++;
			}
		}
		else if (index_Y == imageMatrix.rows - 1) {
			imageArray.at<float>(0, indexImageArray) = imageMatrix.at<float>(index_X, index_Y);
			indexImageArray++;
			index_X++;

			while (index_X < imageMatrix.cols - 1) {
				imageArray.at<float>(0, indexImageArray) = imageMatrix.at<float>(index_X, index_Y);
				indexImageArray++;
				index_Y--;
				index_X++;
			}
		}
	}
	return imageArray;
}

Mat invZigZag(Mat imageArray) {
	Mat imageMatrix = Mat::zeros(sqrt(imageArray.cols), sqrt(imageArray.cols), CV_32F);

	int indexImageArray = 0;

	int index_X = 0;
	int index_Y = 0;

	while (indexImageArray < (imageMatrix.cols * imageMatrix.rows)) {
		if (index_X < imageMatrix.cols - 1) {
			imageMatrix.at<float>(index_X, index_Y) = imageArray.at<float>(0, indexImageArray);
			indexImageArray++;
			index_X++;

			while (index_X > 0) {
				imageMatrix.at<float>(index_X, index_Y) = imageArray.at<float>(0, indexImageArray);
				indexImageArray++;
				index_X--;
				index_Y++;
			}
		}
		else if (index_X == imageMatrix.cols - 1) {
			imageMatrix.at<float>(index_X, index_Y) = imageArray.at<float>(0, indexImageArray);
			indexImageArray++;
			index_Y++;

			while (index_Y < imageMatrix.rows - 1) {
				imageMatrix.at<float>(index_X, index_Y) = imageArray.at<float>(0, indexImageArray);
				indexImageArray++;
				index_X--;
				index_Y++;
			}
		}
		if (index_Y < imageMatrix.rows - 1) {
			imageMatrix.at<float>(index_X, index_Y) = imageArray.at<float>(0, indexImageArray);
			indexImageArray++;
			index_Y++;

			while (index_Y > 0) {
				imageMatrix.at<float>(index_X, index_Y) = imageArray.at<float>(0, indexImageArray);
				indexImageArray++;
				index_Y--;
				index_X++;
			}
		}
		else if (index_Y == imageMatrix.rows - 1) {
			imageMatrix.at<float>(index_X, index_Y) = imageArray.at<float>(0, indexImageArray);
			indexImageArray++;
			index_X++;

			while (index_X < imageMatrix.cols - 1) {
				imageMatrix.at<float>(index_X, index_Y) = imageArray.at<float>(0, indexImageArray);
				indexImageArray++;
				index_Y--;
				index_X++;
			}
		}
	}
	return imageMatrix;
}

Mat encodeRLE(Mat imageArray) {
	Mat imageArrayCoded;

	float _total_rep = 1;
	float _total_nums_non_con = 0;

	for (int i = 0; i < imageArray.cols; i++) {
		float actual;
		float next;

		if (i < imageArray.cols - 2) {
			actual = imageArray.at<float>(0, i);
			next = imageArray.at<float>(0, i + 1);
		}
		else {
			actual = imageArray.at<float>(0, i);
			next = INT_MAX;
		}

		if (actual == next) {
			_total_rep++;
		}
		else {
			_total_nums_non_con++;
			_total_rep = 1;
		}
	}

	imageArrayCoded = Mat(2, _total_nums_non_con, CV_32F, float(0));
	_total_nums_non_con = 0;

	for (int i = 0; i < imageArray.cols; i++) {
		float actual;
		float next;

		if (i < imageArray.cols - 2) {
			actual = imageArray.at<float>(0, i);
			next = imageArray.at<float>(0, i + 1);
		}
		else {
			actual = imageArray.at<float>(0, i);
			next = INT_MAX;
		}

		if (actual == next) {
			_total_rep++;
		}
		else {
			imageArrayCoded.at<float>(0, _total_nums_non_con) = actual;
			imageArrayCoded.at<float>(1, _total_nums_non_con) = _total_rep;
			_total_nums_non_con++;
			_total_rep = 1;
		}
	}

	return imageArrayCoded;
}

Mat decodeRLE(Mat imageArrayCoded) {
	Mat imageArray;

	float _total_rep = 0;

	for (int i = 0; i < imageArrayCoded.cols; i++)
		_total_rep = _total_rep + imageArrayCoded.at<float>(1, i);

	imageArray = Mat(1, _total_rep, CV_32F, float(0));

	int indexImageArray = 0;

	for (int i = 0; i < imageArrayCoded.cols; i++) {
		float val = imageArrayCoded.at<float>(0, i);
		float num_rep = imageArrayCoded.at<float>(1, i);

		for (int j = 0; j < num_rep; j++) {
			imageArray.at<float>(0, indexImageArray) = val;
			indexImageArray++;
		}
	}

	return imageArray;
}

// Define a Huffman Tree Node
struct HuffmanNode {
	float value;
	int frequency;
	HuffmanNode* left;
	HuffmanNode* right;

	HuffmanNode(float val, int freq) : value(val), frequency(freq), left(nullptr), right(nullptr) {}
};

// Comparison object to be used in the priority queue
struct CompareNode {
	bool operator()(HuffmanNode* l, HuffmanNode* r) {
		return l->frequency > r->frequency;
	}
};

// Recursive function to generate Huffman codes from the tree
void generateCodes(HuffmanNode* root, const string& str, unordered_map<float, string>& huffmanCode) {
	if (root == nullptr)
		return;

	// If it's a leaf node, assign the current Huffman code
	if (!root->left && !root->right) {
		huffmanCode[root->value] = str;
	}

	generateCodes(root->left, str + "0", huffmanCode);
	generateCodes(root->right, str + "1", huffmanCode);
}

pair<HuffmanNode*, unordered_map<float, string>> huffmanEncode(const Mat& imageArrayCompressed) {
	// Count the frequency of each value
	unordered_map<float, int> freq;
	for (int i = 0; i < imageArrayCompressed.cols; i++) {
		float value = imageArrayCompressed.at<float>(0, i);
		freq[value]++;
	}

	// Create a priority queue (min-heap)
	priority_queue<HuffmanNode*, vector<HuffmanNode*>, CompareNode> pq;

	// Create a leaf node for each unique value and add it to the priority queue
	for (auto pair : freq) {
		pq.push(new HuffmanNode(pair.first, pair.second));
	}

	// Iterate until the tree is complete
	while (pq.size() != 1) {
		HuffmanNode* left = pq.top(); pq.pop();
		HuffmanNode* right = pq.top(); pq.pop();

		// Create a new internal node with the sum of the frequencies
		int sum = left->frequency + right->frequency;
		HuffmanNode* node = new HuffmanNode(-1, sum);
		node->left = left;
		node->right = right;
		pq.push(node);
	}

	// Root node of the Huffman Tree
	HuffmanNode* root = pq.top();

	// Generate Huffman codes
	unordered_map<float, string> huffmanCode;
	generateCodes(root, "", huffmanCode);

	// Return both the root of the Huffman tree and the Huffman codes
	return { root, huffmanCode };
}

string encodeWithHuffman(const Mat& imageArrayCompressed, const unordered_map<float, string>& huffmanCode) {
	string encodedString = "";

	for (int i = 0; i < imageArrayCompressed.cols; i++) {
		float value = imageArrayCompressed.at<float>(0, i);
		encodedString += huffmanCode.at(value);
	}

	return encodedString;
}

Mat decodeHuffman(const string& encodedData, HuffmanNode* root, int originalSize) {
	Mat decodedData(1, originalSize, CV_32F);
	HuffmanNode* currentNode = root;
	int index = 0;

	for (char bit : encodedData) {
		if (bit == '0')
			currentNode = currentNode->left;
		else
			currentNode = currentNode->right;

		if (!currentNode->left && !currentNode->right) {
			decodedData.at<float>(0, index++) = currentNode->value;
			currentNode = root;
		}
	}

	return decodedData;
}

void MultDerechaCPU(float* const imagenOriginal, float* const imagenSalida,
	float* const mascara, int filas, int columnas, bool setQuantization) {
	for (int fila = 0; fila < filas; fila += DIM_MASCARA) {
		for (int columna = 0; columna < columnas; columna += DIM_MASCARA) {
			for (int r = 0; r < DIM_MASCARA; r++) {
				for (int c = 0; c < DIM_MASCARA; c++) {
					float total = 0;
					for (int k = 0; k < DIM_MASCARA; k++) {
						int indMascara = (r * DIM_MASCARA) + k;
						int indImagen = ((fila + k) * columnas) + (c + columna);
						if (setQuantization)
							total = total + mascara[indMascara] * imagenOriginal[indImagen];
						else
							total = total + mascara[indMascara] * (imagenOriginal[indImagen] * DELTA);
					}
					int indice = (fila + r) * columnas + (columna + c);
					imagenSalida[indice] = total;
				}
			}
		}
	}
}

void MultIzquierdaCPU(float* const imagenOriginal, float* const imagenSalida,
	float* const mascara, int filas, int columnas, bool setQuantization) {
	for (int fila = 0; fila < filas; fila += DIM_MASCARA) {
		for (int columna = 0; columna < columnas; columna += DIM_MASCARA) {
			for (int r = 0; r < DIM_MASCARA; r++) {
				for (int c = 0; c < DIM_MASCARA; c++) {
					float total = 0;
					for (int k = 0; k < DIM_MASCARA; k++) {
						int indMascara = (k * DIM_MASCARA) + c;
						int indImagen = ((fila + r) * filas) + (k + columna);
						total = total + mascara[indMascara] * imagenOriginal[indImagen];
					}
					int indice = (fila + r) * columnas + (columna + c);
					if (setQuantization)
						imagenSalida[indice] = round(total / DELTA);
					else
						imagenSalida[indice] = total;
				}
			}
		}
	}
}

void MultDerechaCPU_OpenMP(float* const imagenOriginal, float* const imagenSalida,
	float* const mascara, int filas, int columnas, bool setQuantization) {

	// Parallelize the outer loops and work on independent chunks of the image
#pragma omp parallel for collapse(2) schedule(static)
	for (int fila = 0; fila < filas; fila += DIM_MASCARA) {
		for (int columna = 0; columna < columnas; columna += DIM_MASCARA) {
			for (int r = 0; r < DIM_MASCARA; r++) {
				for (int c = 0; c < DIM_MASCARA; c++) {
					float total = 0;
					for (int k = 0; k < DIM_MASCARA; k++) {
						int indMascara = (r * DIM_MASCARA) + k;
						int indImagen = ((fila + k) * columnas) + (c + columna);
						total += mascara[indMascara] * (setQuantization ? imagenOriginal[indImagen] : imagenOriginal[indImagen] * DELTA);
					}
					int indice = (fila + r) * columnas + (columna + c);

					// Write to imagenSalida independently; no need for critical section
					imagenSalida[indice] = total;
				}
			}
		}
	}
}

void MultIzquierdaCPU_OpenMP(float* const imagenOriginal, float* const imagenSalida,
	float* const mascara, int filas, int columnas, bool setQuantization) {

	// Parallelize the outer loops and work on independent chunks of the image
#pragma omp parallel for collapse(2) schedule(static)
	for (int fila = 0; fila < filas; fila += DIM_MASCARA) {
		for (int columna = 0; columna < columnas; columna += DIM_MASCARA) {
			for (int r = 0; r < DIM_MASCARA; r++) {
				for (int c = 0; c < DIM_MASCARA; c++) {
					float total = 0;
					for (int k = 0; k < DIM_MASCARA; k++) {
						int indMascara = (k * DIM_MASCARA) + c;
						int indImagen = ((fila + r) * filas) + (k + columna);
						total += mascara[indMascara] * imagenOriginal[indImagen];
					}
					int indice = (fila + r) * columnas + (columna + c);

					// Write to imagenSalida independently; no need for critical section
					imagenSalida[indice] = setQuantization ? round(total / DELTA) : total;
				}
			}
		}
	}
}

void MultDerechaCPU_OpenMP2(float* const imagenOriginal, float* const imagenSalida,
	float* const mascara, int filas, int columnas, bool setQuantization) {

	auto func = [=](const cv::Range& range) {
		for (int fila = range.start; fila < range.end; fila += DIM_MASCARA) {
			for (int columna = 0; columna < columnas; columna += DIM_MASCARA) {
				for (int r = 0; r < DIM_MASCARA; r++) {
					for (int c = 0; c < DIM_MASCARA; c++) {
						float total = 0;
#pragma omp simd reduction(+:total)
						for (int k = 0; k < DIM_MASCARA; k++) {
							int indMascara = r * DIM_MASCARA + k;
							int indImagen = (fila + k) * columnas + (c + columna);
							float imgVal = imagenOriginal[indImagen];
							total += mascara[indMascara] * (setQuantization ? imgVal : imgVal * DELTA);
						}
						int indice = (fila + r) * columnas + (columna + c);
						imagenSalida[indice] = total;
					}
				}
			}
		}
	};

	cv::parallel_for_(cv::Range(0, filas), func, cv::getNumThreads());
}

void MultIzquierdaCPU_OpenMP2(float* const imagenOriginal, float* const imagenSalida,
	float* const mascara, int filas, int columnas, bool setQuantization) {

	auto func = [=](const cv::Range& range) {
		for (int fila = range.start; fila < range.end; fila += DIM_MASCARA) {
			for (int columna = 0; columna < columnas; columna += DIM_MASCARA) {
				for (int r = 0; r < DIM_MASCARA; r++) {
					for (int c = 0; c < DIM_MASCARA; c++) {
						float total = 0;
#pragma omp simd reduction(+:total)
						for (int k = 0; k < DIM_MASCARA; k++) {
							int indMascara = k * DIM_MASCARA + c;
							int indImagen = (fila + r) * columnas + (k + columna);
							total += mascara[indMascara] * imagenOriginal[indImagen];
						}
						int indice = (fila + r) * columnas + (columna + c);
						imagenSalida[indice] = setQuantization ? round(total / DELTA) : total;
					}
				}
			}
		}
	};

	cv::parallel_for_(cv::Range(0, filas), func, cv::getNumThreads());
}

void executeMultDerechaOpenCL(cl_context context, cl_command_queue queue, cl_program program,
	float* imagenOriginal, float* imagenSalida, float* mascara,
	int filas, int columnas, bool setQuantization) {
	cl_int err;

	// Create buffers for imagenOriginal, imagenSalida, and mascara
	cl_mem d_imagenOriginal = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		filas * columnas * sizeof(float), imagenOriginal, &err);
	cl_mem d_imagenSalida = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
		filas * columnas * sizeof(float), NULL, &err);
	cl_mem d_mascara = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		DIM_MASCARA * DIM_MASCARA * sizeof(float), mascara, &err);

	// Create kernel
	cl_kernel kernel = clCreateKernel(program, "MultDerechaOpenCL", &err);

	// Set kernel arguments
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_imagenOriginal);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_imagenSalida);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_mascara);
	clSetKernelArg(kernel, 3, sizeof(int), &filas);
	clSetKernelArg(kernel, 4, sizeof(int), &columnas);
	clSetKernelArg(kernel, 5, sizeof(int), &setQuantization);

	// Define global and local work sizes
	size_t globalWorkSize = filas * columnas;

	// Enqueue the kernel for execution
	clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalWorkSize, NULL, 0, NULL, NULL);

	// Wait for execution to complete
	clFinish(queue);

	// Read back the result
	clEnqueueReadBuffer(queue, d_imagenSalida, CL_TRUE, 0, filas * columnas * sizeof(float), imagenSalida, 0, NULL, NULL);

	// Clean up
	clReleaseMemObject(d_imagenOriginal);
	clReleaseMemObject(d_imagenSalida);
	clReleaseMemObject(d_mascara);
	clReleaseKernel(kernel);
}

void executeMultIzquierdaOpenCL(cl_context context, cl_command_queue queue, cl_program program,
	float* imagenOriginal, float* imagenSalida, float* mascara,
	int filas, int columnas, bool setQuantization) {
	cl_int err;

	// Create buffers for imagenOriginal, imagenSalida, and mascara
	cl_mem d_imagenOriginal = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		filas * columnas * sizeof(float), imagenOriginal, &err);
	cl_mem d_imagenSalida = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
		filas * columnas * sizeof(float), NULL, &err);
	cl_mem d_mascara = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		DIM_MASCARA * DIM_MASCARA * sizeof(float), mascara, &err);

	// Create kernel
	cl_kernel kernel = clCreateKernel(program, "MultIzquierdaOpenCL", &err);

	// Set kernel arguments
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_imagenOriginal);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_imagenSalida);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_mascara);
	clSetKernelArg(kernel, 3, sizeof(int), &filas);
	clSetKernelArg(kernel, 4, sizeof(int), &columnas);
	clSetKernelArg(kernel, 5, sizeof(int), &setQuantization);

	// Define global and local work sizes
	size_t globalWorkSize = filas * columnas;

	// Enqueue the kernel for execution
	clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalWorkSize, NULL, 0, NULL, NULL);

	// Wait for execution to complete
	clFinish(queue);

	// Read back the result
	clEnqueueReadBuffer(queue, d_imagenSalida, CL_TRUE, 0, filas * columnas * sizeof(float), imagenSalida, 0, NULL, NULL);

	// Clean up
	clReleaseMemObject(d_imagenOriginal);
	clReleaseMemObject(d_imagenSalida);
	clReleaseMemObject(d_mascara);
	clReleaseKernel(kernel);
}

uintmax_t getFileSize(const string& path) {
	return filesystem::file_size(path);
}

void saveCompressedImage(const string& path, int rows, int cols, int type, const string& huffmanEncodedString) {
	// Open file in binary mode
	ofstream outputFile(path, ios::binary);
	if (outputFile.is_open()) {
		// JFIF-like file signature (e.g., "MYJFIF")
		string fileSignature = "GOKGOKGOK";
		outputFile.write(fileSignature.c_str(), fileSignature.size());

		// Metadata (image dimensions, type)
		outputFile.write(reinterpret_cast<char*>(&rows), sizeof(rows));
		outputFile.write(reinterpret_cast<char*>(&cols), sizeof(cols));
		outputFile.write(reinterpret_cast<char*>(&type), sizeof(type));

		// Compressed data size
		int dataSize = huffmanEncodedString.size();
		outputFile.write(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));

		// Write the Huffman-encoded data
		outputFile.write(huffmanEncodedString.c_str(), dataSize);

		// End marker (optional)
		string endMarker = "ENDGOKGOK";
		outputFile.write(endMarker.c_str(), endMarker.size());

		outputFile.close();
		cout << "Compressed data saved to: " << path << endl;
	}
	else {
		cout << "Failed to save compressed data." << endl;
	}
}

void loadCompressedImage(const string& filename, int& rows, int& cols, int& type, string& huffmanEncodedString) {
	ifstream file(filename, ios::in | ios::binary);
	if (!file.is_open()) {
		cerr << "Error opening file for reading!" << endl;
		return;
	}

	// Read and validate the file signature
	char signature[9];
	file.read(signature, 9);
	if (string(signature, 9) != "GOKGOKGOK") {
		cerr << "Invalid file format!" << endl;
		return;
	}

	// Read metadata
	file.read(reinterpret_cast<char*>(&rows), sizeof(rows));  // Image rows
	file.read(reinterpret_cast<char*>(&cols), sizeof(cols));  // Image columns
	file.read(reinterpret_cast<char*>(&type), sizeof(type));  // Image type

	// Read Huffman-encoded data size
	int dataSize;
	file.read(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));

	// Read Huffman-encoded data
	huffmanEncodedString.resize(dataSize);
	file.read(&huffmanEncodedString[0], dataSize);

	// Validate end marker
	char endMarker[9];
	file.read(endMarker, 9);
	if (string(endMarker, 9) != "ENDGOKGOK") {
		cerr << "Invalid file end marker!" << endl;
	}

	file.close();
}

void drawBarChart(const vector<double>& executionTimes, const vector<string>& labels) {
	// Parameters
	int width = 800;
	int height = 600;
	int margin = 50;
	int barWidth = 100;
	int barSpacing = 100;
	int yAxisInterval = 10;

	// Create a blank image with white background
	Mat chart(height, width, CV_8UC3, Scalar(255, 255, 255));

	// Find the maximum execution time for normalization
	double maxTime = *max_element(executionTimes.begin(), executionTimes.end());
	if (maxTime > 500)
		yAxisInterval = 100;
	else
		yAxisInterval = 10;
	double roundedMaxTime = ceil(maxTime / yAxisInterval) * yAxisInterval;

	// Find the index of the best performance (minimum time)
	auto minElementIter = min_element(executionTimes.begin(), executionTimes.end());
	size_t bestIndex = distance(executionTimes.begin(), minElementIter);

	// Draw bars
	for (size_t i = 0; i < executionTimes.size(); ++i) {
		int barHeight = static_cast<int>((executionTimes[i] / roundedMaxTime) * (height - 2 * margin)) ;
		int x = margin + i * (barWidth + barSpacing);
		int y = height - margin - barHeight;
		Rect barRect(x, y, barWidth, barHeight);
		rectangle(chart, barRect, Scalar(0, 0, 255), -1); // Red bars
	}

	// Draw X and Y axes
	line(chart, Point(margin, height - margin), Point(width - margin, height - margin), Scalar(0, 0, 0), 2);
	line(chart, Point(margin, margin), Point(margin, height - margin), Scalar(0, 0, 0), 2);

	putText(chart, "Execution Time (ms)", Point(20, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);

	double adjustedTime;
	if ((int)maxTime % yAxisInterval != 0)
		adjustedTime = maxTime + yAxisInterval;
	else
		adjustedTime = maxTime;

	// Draw labels y
	int yAxisLabelsCount = static_cast<int>((adjustedTime) / yAxisInterval) + 1;
	for (int i = 0; i < yAxisLabelsCount; ++i) {
		int yLabel = height - margin - (i * (height - 2 * margin) / (yAxisLabelsCount - 1));
		int yValue = yAxisInterval * i;
		putText(chart, to_string(yValue) + "ms", Point(margin - 40, yLabel + 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
	}

	// Draw labels x
	for (size_t i = 0; i < labels.size(); ++i) {
		int x = margin + i * (barWidth + barSpacing) + barWidth / 4;
		int y = height - margin + 20; // Place label below the X axis
		putText(chart, labels[i], Point(x - 10, y), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
	}

	// Add text for the best performance with execution time in two lines
	int bestX = margin + bestIndex * (barWidth + barSpacing) + barWidth / 4;
	int bestY = height - margin - (executionTimes[bestIndex] / roundedMaxTime) * (height - 2 * margin);
	string bestPerformanceText1 = "Best Performance:";
	string bestPerformanceText2 = to_string((int)executionTimes[bestIndex]) + " ms";
	putText(chart, bestPerformanceText1, Point(bestX - 70, bestY - 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
	putText(chart, bestPerformanceText2, Point(bestX, bestY - 10), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);

	// Display the image
	imshow("Bar Chart", chart);
	waitKey(0);
}

void chck() {
	cout << "OpenCV version: " << CV_VERSION << endl;
	cout << "Available image formats: ";
	vector <String> formats;
	cout << getBuildInformation() << endl;
}

double mainCPU(const string nombreImagen) {
	Mat imagen;
	bool setQuantization;
	float* imagenOriginal;
	imagen = imread(nombreImagen.c_str(), IMREAD_GRAYSCALE);
	imagen.convertTo(imagen, CV_32F);

	if (imagen.empty()) {
		cout << "Empty Image...";
	}
	int filas = imagen.rows;
	int columnas = imagen.cols;
	const size_t numPixeles = filas * columnas;
	float mascara[LONG_MASCARA] = {
		0.3536, 0.3536, 0.5000, 0, 0.7071, 0, 0, 0,
		0.3536, 0.3536, 0.5000, 0, -0.7071, 0, 0, 0,
		0.3536, 0.3536, -0.5000, 0, 0, 0.7071, 0, 0,
		0.3536, 0.3536, -0.5000, 0, 0, -0.7071, 0, 0,
		0.3536, -0.3536, 0, 0.5000, 0, 0, 0.7071, 0,
		0.3536, -0.3536, 0, 0.5000, 0, 0, -0.7071, 0,
		0.3536, -0.3536, 0, -0.5000, 0, 0, 0, 0.7071,
		0.3536, -0.3536, 0, -0.5000, 0, 0, 0, -0.7071
	};
	float mascara_inv[LONG_MASCARA] = {
		0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536,
		0.3536, 0.3536, 0.3536, 0.3536, -0.3536, -0.3536, -0.3536, -0.3536,
		0.5000, 0.5000, -0.5000, -0.5000, 0, 0, 0, 0,
		0, 0, 0, 0, 0.5000, 0.5000, -0.5000, -0.5000,
		0.7071, -0.7071, 0, 0, 0, 0, 0, 0,
		0, 0, 0.7071, -0.7071, 0, 0, 0, 0,
		0, 0, 0, 0, 0.7071, -0.7071, 0, 0,
		0, 0, 0, 0, 0, 0, 0.7071, -0.7071
	};
	imagenOriginal = (float*)imagen.ptr<float>(0);
	float* imagenTransf1 = new float[numPixeles];
	float* imagenTransf2 = new float[numPixeles];
	float* imagenTransf3 = new float[numPixeles];
	float* imagenTransf4 = new float[numPixeles];

	clock_t timer1 = clock();
	setQuantization = true;

	int newRow = filas;
	int newCol = columnas;
	int type = imagen.type();

	while (newRow > TARGET_SIZE && newCol > TARGET_SIZE) {
		// Apply the Haar wavelet transform
		MultDerechaCPU(imagenOriginal, imagenTransf1, mascara_inv, newRow, newCol, setQuantization);
		MultIzquierdaCPU(imagenTransf1, imagenTransf2, mascara, newRow, newCol, setQuantization);

		// Halve the rows and columns for the next iteration
		newRow /= 2;
		newCol /= 2;

		// Update imagenOriginal for the next iteration
		imagenOriginal = imagenTransf2;
	}
	MultDerechaCPU(imagenTransf2, imagenTransf3, mascara_inv, newRow, newCol, setQuantization);
	MultIzquierdaCPU(imagenTransf3, imagenTransf4, mascara, newRow, newCol, setQuantization);

	Mat imagenWaveletTransformadaCPU(newRow, newCol, CV_32F, imagenTransf4);
	Mat ImageArray = toZigZag(imagenWaveletTransformadaCPU);
	Mat ImageArrayCompressed = encodeRLE(ImageArray);
	// Get the Huffman tree and the codes
	auto [root, huffmanCode] = huffmanEncode(ImageArrayCompressed);

	// Encode the image data using the Huffman codes
	string huffmanEncodedString = encodeWithHuffman(ImageArrayCompressed, huffmanCode);

	timer1 = clock() - timer1;

	cout << "\n\nCompressing with CPU" << endl;
	string path = "../../Compressed_Image/compressedDataCPU.bin";
	saveCompressedImage(path, newRow, newCol, imagen.type(), huffmanEncodedString);

	int InputBitcost = imagen.rows * imagen.cols * 8;
	cout << "InputBitcost: " << InputBitcost << endl;
	float OutputBitcost = huffmanEncodedString.length();
	cout << "OutputBitcost: " << OutputBitcost << endl;
	uintmax_t originalFileSize = getFileSize(nombreImagen);
	cout << "Original image file size: " << originalFileSize << " bytes" << endl;
	uintmax_t compressedFileSize = filesystem::file_size(path);
	cout << "Compressed file size: " << compressedFileSize << " bytes" << endl;
	cout << "Compression Ratio: " << originalFileSize / (double)compressedFileSize << endl;

	String binaryString;
	loadCompressedImage(path, newRow, newCol, type, binaryString);
	// Decode the Huffman-encoded string using the root node
	ImageArray = decodeHuffman(binaryString, root, ImageArrayCompressed.cols);

	ImageArray = decodeRLE(ImageArrayCompressed);
	imagenWaveletTransformadaCPU = invZigZag(ImageArray);

	float* tempTransf = (float*)imagenWaveletTransformadaCPU.ptr<float>(0);

	setQuantization = false;
	while (newRow < filas && newCol < columnas) {
		// Apply the Haar wavelet transform
		MultDerechaCPU(tempTransf, imagenTransf1, mascara, newRow, newCol, setQuantization);
		MultIzquierdaCPU(imagenTransf1, imagenTransf2, mascara_inv, newRow, newCol, setQuantization);

		// Halve the rows and columns for the next iteration
		newRow *= 2;
		newCol *= 2;

		// Update imagenOriginal for the next iteration
		tempTransf = imagenTransf2;
	}
	MultDerechaCPU(imagenTransf2, imagenTransf3, mascara, newRow, newCol, setQuantization);
	MultIzquierdaCPU(imagenTransf3, imagenTransf4, mascara_inv, newRow, newCol, setQuantization);

	//timer1 = clock() - timer1;

	printf("Image Size: [%d, %d]\n", filas, columnas);
	printf("CPU Execution Time is %10.3f ms.\n", ((timer1) / double(CLOCKS_PER_SEC) * 1000));

	Mat imagenRecuperada(filas, columnas, CV_32F, imagenTransf4);

	if (filas > 1024 && columnas > 1024)
	{
		double scale = 1024 / (double)filas;
		// Resize the image to fit the window
		resize(imagen, imagen, Size(), scale, scale, INTER_LINEAR);
		resize(imagenRecuperada, imagenRecuperada, Size(), scale, scale, INTER_LINEAR);
	}

	imagenRecuperada.convertTo(imagenRecuperada, CV_64F);
	imagen.convertTo(imagen, CV_64F);

	// ------ IMAGE METRICS -----
	cout << "------IMAGE METRICS-----" << endl;
	// CR
	cout << "The CR value is: " << InputBitcost / OutputBitcost << endl;
	// MSE
	cout << "The MSE value is: " << qm::eqm(imagen, imagenRecuperada) << endl;
	// PSNR
	cout << "The PSNR value is: " << qm::psnr(imagen, imagenRecuperada, 1) << endl;
	// SSIM
	//cout << "The SSIM value is: " << qm::ssim(imagen, imagenRecuperada, 1) << endl;

	imagenRecuperada.convertTo(imagenRecuperada, CV_8UC1);
	imagen.convertTo(imagen, CV_8UC1);

	imshow("Original", imagen);
	imshow("Compressed on CPU", imagenRecuperada);

	delete[] imagenTransf1;
	delete[] imagenTransf2;
	delete[] imagenTransf3;
	delete[] imagenTransf4;

	return timer1;
}

double mainGPU(const string nombreImagen)
{
	cudaFree(0);

	Mat imagen;
	bool setQuantization;

	float* imagenOriginal, * dImagenOriginal, * dMascara;
	float* dimagenTransf1, * dimagenTransf2, * dimagenTransf3, * dimagenTransf4;
	float* imagenTransf1GPU;

	imagen = imread(nombreImagen.c_str(), IMREAD_GRAYSCALE);
	imagen.convertTo(imagen, CV_32F);

	if (imagen.empty()) {
		cout << "Empty Image...";
	}

	int filas = imagen.rows;
	int columnas = imagen.cols;
	const size_t numPixeles = filas * columnas;

	float mascara[LONG_MASCARA] =
	{
		0.3536, 0.3536, 0.5000, 0, 0.7071, 0, 0, 0,
		0.3536, 0.3536, 0.5000, 0, -0.7071, 0, 0, 0,
		0.3536, 0.3536, -0.5000, 0, 0, 0.7071, 0, 0,
		0.3536, 0.3536, -0.5000, 0, 0, -0.7071, 0, 0,
		0.3536, -0.3536, 0, 0.5000, 0, 0, 0.7071, 0,
		0.3536, -0.3536, 0, 0.5000, 0, 0, -0.7071, 0,
		0.3536, -0.3536, 0, -0.5000, 0, 0, 0, 0.7071,
		0.3536, -0.3536, 0, -0.5000, 0, 0, 0, -0.7071
	};

	float mascara_inv[LONG_MASCARA] =
	{
		0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536,
		0.3536, 0.3536, 0.3536, 0.3536, -0.3536, -0.3536, -0.3536, -0.3536,
		0.5000, 0.5000, -0.5000, -0.5000, 0, 0, 0, 0,
		0, 0, 0, 0, 0.5000, 0.5000, -0.5000, -0.5000,
		0.7071, -0.7071, 0, 0, 0, 0, 0, 0,
		0, 0, 0.7071, -0.7071, 0, 0, 0, 0,
		0, 0, 0, 0, 0.7071, -0.7071, 0, 0,
		0, 0, 0, 0, 0, 0, 0.7071, -0.7071
	};

	imagenOriginal = (float*)imagen.ptr<float>(0);
	imagenTransf1GPU = (float*)malloc(sizeof(float) * numPixeles);

	int N = DIM_MASCARA, M = DIM_MASCARA;
	int newRow = filas;
	int newCol = columnas;
	int type = imagen.type();

	const dim3 gridSize(filas / M, columnas / N, 1);
	const dim3 blockSize(M, N, 1);

	cudaMalloc(&dImagenOriginal, sizeof(float) * numPixeles);
	cudaMalloc(&dMascara, sizeof(float) * LONG_MASCARA);
	cudaMalloc(&dimagenTransf1, sizeof(float) * numPixeles);
	cudaMalloc(&dimagenTransf2, sizeof(float) * numPixeles);
	cudaMalloc(&dimagenTransf3, sizeof(float) * numPixeles);
	cudaMalloc(&dimagenTransf4, sizeof(float) * numPixeles);

	cudaMemcpy(dImagenOriginal, imagenOriginal, sizeof(float) * numPixeles, cudaMemcpyHostToDevice);

	clock_t timer1 = clock();

	setQuantization = true;
	dim3 reducedGridSize(newRow / M, newCol / N, 1);

	while (newRow > TARGET_SIZE && newCol > TARGET_SIZE) {
		// First step: MultDerechaGPU with inverse mask
		cudaMemcpy(dMascara, mascara_inv, sizeof(float) * LONG_MASCARA, cudaMemcpyHostToDevice);
		HaarWaveletWrapper::MultDerechaGPU(reducedGridSize, blockSize, dImagenOriginal, dimagenTransf1, dMascara, newRow, newCol, setQuantization);

		// Second step: MultIzquierdaGPU with forward mask
		cudaMemcpy(dMascara, mascara, sizeof(float) * LONG_MASCARA, cudaMemcpyHostToDevice);
		HaarWaveletWrapper::MultIzquierdaGPU(reducedGridSize, blockSize, dimagenTransf1, dimagenTransf2, dMascara, newRow, newCol, setQuantization);

		// Reduce image size for the next iteration
		newRow /= 2;
		newCol /= 2;

		// Update grid size for the next level
		reducedGridSize.x = newRow / M;
		reducedGridSize.y = newCol / N;
		reducedGridSize.z = 1;

		// Copy dimagenTransf2 back into dImagenOriginal
		cudaMemcpy(dImagenOriginal, dimagenTransf2, sizeof(float) * newRow * newCol, cudaMemcpyDeviceToDevice);
	}

	cudaMemcpy(dMascara, mascara_inv, sizeof(float) * LONG_MASCARA, cudaMemcpyHostToDevice);
	HaarWaveletWrapper::MultDerechaGPU(reducedGridSize, blockSize, dimagenTransf2, dimagenTransf3, dMascara, newRow, newCol, setQuantization);
	cudaMemcpy(dMascara, mascara, sizeof(float) * LONG_MASCARA, cudaMemcpyHostToDevice);
	HaarWaveletWrapper::MultIzquierdaGPU(reducedGridSize, blockSize, dimagenTransf3, dimagenTransf4, dMascara, newRow, newCol, setQuantization);

	cudaDeviceSynchronize();
	cudaMemcpy(imagenTransf1GPU, dimagenTransf4, sizeof(float) * numPixeles, cudaMemcpyDeviceToHost);

	Mat imagenWaveletTransformadaGPU(newRow, newCol, CV_32F, imagenTransf1GPU);
	Mat ImageArray = toZigZag(imagenWaveletTransformadaGPU);
	Mat ImageArrayCompressed = encodeRLE(ImageArray);

	// Get the Huffman tree and the codes
	auto [root, huffmanCode] = huffmanEncode(ImageArrayCompressed);

	// Encode the image data using the Huffman codes
	string huffmanEncodedString = encodeWithHuffman(ImageArrayCompressed, huffmanCode);

	timer1 = clock() - timer1;

	cout << "\n\nCompressing with GPU" << endl;
	string path = "../../Compressed_Image/compressedDataGPU.bin";
	saveCompressedImage(path, newRow, newCol, imagen.type(), huffmanEncodedString);

	int InputBitcost = imagen.rows * imagen.cols * 8;
	cout << "InputBitcost: " << InputBitcost << endl;

	float OutputBitcost = huffmanEncodedString.length();
	cout << "OutputBitcost: " << OutputBitcost << endl;

	// Display file sizes
	uintmax_t originalFileSize = getFileSize(nombreImagen);
	cout << "Original image file size: " << originalFileSize << " bytes" << endl;
	uintmax_t compressedFileSize = filesystem::file_size(path);
	cout << "Compressed file size: " << compressedFileSize << " bytes" << endl;
	cout << "Compression Ratio: " << originalFileSize / (double)compressedFileSize << endl;

	// ------ END -------
	String binaryString;
	loadCompressedImage(path, newRow, newCol, type, binaryString);
	// Decode Huffman
	ImageArray = decodeHuffman(binaryString, root, ImageArrayCompressed.cols);
	ImageArray = decodeRLE(ImageArrayCompressed);
	imagenWaveletTransformadaGPU = invZigZag(ImageArray);
	cudaMemcpy(dImagenOriginal, (float*)imagenWaveletTransformadaGPU.ptr<float>(0), sizeof(float) * newRow * newCol, cudaMemcpyHostToDevice);

	setQuantization = false;

	while (newRow < filas && newCol < columnas) {
		// Apply the Haar wavelet transform
		cudaMemcpy(dMascara, mascara, sizeof(float) * LONG_MASCARA, cudaMemcpyHostToDevice);
		HaarWaveletWrapper::MultDerechaGPU(reducedGridSize, blockSize, dImagenOriginal, dimagenTransf1, dMascara, newRow, newCol, setQuantization);
		cudaMemcpy(dMascara, mascara_inv, sizeof(float) * LONG_MASCARA, cudaMemcpyHostToDevice);
		HaarWaveletWrapper::MultIzquierdaGPU(reducedGridSize, blockSize, dimagenTransf1, dimagenTransf2, dMascara, newRow, newCol, setQuantization);

		newRow *= 2;
		newCol *= 2;
		reducedGridSize.x = newRow / M;
		reducedGridSize.y = newCol / N;
		reducedGridSize.z = 1;

		// Copy dimagenTransf2 back into dImagenOriginal
		cudaMemcpy(dImagenOriginal, dimagenTransf2, sizeof(float) * newRow * newCol, cudaMemcpyDeviceToDevice);
	}
	cudaMemcpy(dMascara, mascara, sizeof(float) * LONG_MASCARA, cudaMemcpyHostToDevice);
	HaarWaveletWrapper::MultDerechaGPU(reducedGridSize, blockSize, dimagenTransf2, dimagenTransf3, dMascara, newRow, newCol, setQuantization);
	cudaMemcpy(dMascara, mascara_inv, sizeof(float) * LONG_MASCARA, cudaMemcpyHostToDevice);
	HaarWaveletWrapper::MultIzquierdaGPU(reducedGridSize, blockSize, dimagenTransf3, dimagenTransf4, dMascara, newRow, newCol, setQuantization);

	cudaDeviceSynchronize();
	cudaMemcpy(imagenTransf1GPU, dimagenTransf4, sizeof(float) * numPixeles, cudaMemcpyDeviceToHost);

	//timer1 = clock() - timer1;

	printf("Image Size: [%d, %d]\n", filas, columnas);
	printf("GPU Execution Time is %10.3f ms.\n", ((timer1) / double(CLOCKS_PER_SEC) * 1000));

	Mat imagenRecuperada(filas, columnas, CV_32F, imagenTransf1GPU);

	if (filas > 1024 && columnas > 1024)
	{
		double scale = 1024 / (double)filas;
		// Resize the image to fit the window
		resize(imagen, imagen, Size(), scale, scale, INTER_LINEAR);
		resize(imagenRecuperada, imagenRecuperada, Size(), scale, scale, INTER_LINEAR);
	}

	imagenRecuperada.convertTo(imagenRecuperada, CV_64F);
	imagen.convertTo(imagen, CV_64F);

	// ------ IMAGE METRICS -----
	cout << "------IMAGE METRICS-----" << endl;
	// CR
	cout << "The CR value is: " << InputBitcost / OutputBitcost << endl;
	// MSE
	cout << "The MSE value is: " << qm::eqm(imagen, imagenRecuperada) << endl;
	// PSNR
	cout << "The PSNR value is: " << qm::psnr(imagen, imagenRecuperada, 1) << endl;
	// SSIM
	//cout << "The SSIM value is: " << qm::ssim(imagen, imagenRecuperada, 1) << endl;

	imagenRecuperada.convertTo(imagenRecuperada, CV_8UC1);
	imagen.convertTo(imagen, CV_8UC1);

	imshow("Original", imagen);
	imshow("Compressed in GPU", imagenRecuperada);

	return timer1;
};

double mainOpenCL(const string nombreImagen) {
	// Step 1: Initialize OpenCL
	cl_int err;
	cl_platform_id platform = NULL;
	cl_device_id device = NULL;
	cl_context context = NULL;
	cl_command_queue queue = NULL;
	cl_program program = NULL;

	// Get platform and device information
	err = clGetPlatformIDs(1, &platform, NULL);
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

	// Create OpenCL context and command queue
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	queue = clCreateCommandQueue(context, device, 0, &err);

	// Create and build the OpenCL program
	// Define the kernel source as a string
	const char* kernelSource =
		"const int DELTA = 1;\n"
		"const int DIM_MASCARA = 8;  // Define mask size\n "
		"__kernel void MultDerechaOpenCL(__global const float* imagenOriginal, __global float* imagenSalida, "
		"    __global const float* mascara, int filas, int columnas, int setQuantization) { "
		"    int fila = get_global_id(0) / columnas;   // Calculate row\n "
		"    int columna = get_global_id(0) % columnas; // Calculate column\n "
		"    if (fila % DIM_MASCARA == 0 && columna % DIM_MASCARA == 0 && fila + DIM_MASCARA <= filas && columna + DIM_MASCARA <= columnas) { "
		"        for (int r = 0; r < DIM_MASCARA; r++) { "
		"            for (int c = 0; c < DIM_MASCARA; c++) { "
		"                float total = 0.0f;\n "
		"                for (int k = 0; k < DIM_MASCARA; k++) { "
		"                    int indMascara = (r * DIM_MASCARA) + k;\n "
		"                    int indImagen = ((fila + k) * columnas) + (c + columna);\n "
		"                    if (setQuantization) \n"
		"                        total += mascara[indMascara] * imagenOriginal[indImagen]; \n"
		"                    else\n "
		"                        total += mascara[indMascara] * (imagenOriginal[indImagen] * DELTA);\n "
		"                }\n "
		"                int indice = (fila + r) * columnas + (columna + c);\n "
		"                imagenSalida[indice] = total;\n "
		"            }\n "
		"        }\n "
		"    }\n "
		"}\n "

		"__kernel void MultIzquierdaOpenCL(__global const float* imagenOriginal, __global float* imagenSalida, "
		"    __global const float* mascara, int filas, int columnas, int setQuantization) { "
		"    int fila = get_global_id(0) / columnas;   // Calculate row\n "
		"    int columna = get_global_id(0) % columnas; // Calculate column\n "
		"    if (fila % DIM_MASCARA == 0 && columna % DIM_MASCARA == 0 && fila + DIM_MASCARA <= filas && columna + DIM_MASCARA <= columnas) { "
		"        for (int r = 0; r < DIM_MASCARA; r++) { "
		"            for (int c = 0; c < DIM_MASCARA; c++) { "
		"                float total = 0.0f;\n "
		"                for (int k = 0; k < DIM_MASCARA; k++) { "
		"                    int indMascara = (k * DIM_MASCARA) + c;\n "
		"                    int indImagen = ((fila + r) * filas) + (k + columna);\n "
		"                    total += mascara[indMascara] * imagenOriginal[indImagen];\n "
		"                }\n "
		"                int indice = (fila + r) * columnas + (columna + c);\n "
		"                if (setQuantization)\n "
		"                    imagenSalida[indice] = round(total / DELTA);\n "
		"                else\n "
		"                    imagenSalida[indice] = total;\n "
		"            }\n "
		"        }\n "
		"    }\n "
		"}";

	// Create and build the OpenCL program
	program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
	err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

	if (err != CL_SUCCESS) {
		// Get the size of the log
		size_t log_size;
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

		// Allocate memory for the log
		char* log = (char*)malloc(log_size);

		// Get the log
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

		// Print the log
		printf("Build log:\n%s\n", log);
		free(log);
	}

	// Step 2: Load Image and Convert
	Mat imagen = imread(nombreImagen.c_str(), IMREAD_GRAYSCALE);
	bool setQuantization;
	imagen.convertTo(imagen, CV_32F);
	if (imagen.empty()) {
		cout << "Empty Image...";
		return -1;
	}

	int filas = imagen.rows;
	int columnas = imagen.cols;
	size_t numPixeles = filas * columnas;

	float mascara[LONG_MASCARA] = {
		0.3536, 0.3536, 0.5000, 0, 0.7071, 0, 0, 0,
		0.3536, 0.3536, 0.5000, 0, -0.7071, 0, 0, 0,
		0.3536, 0.3536, -0.5000, 0, 0, 0.7071, 0, 0,
		0.3536, 0.3536, -0.5000, 0, 0, -0.7071, 0, 0,
		0.3536, -0.3536, 0, 0.5000, 0, 0, 0.7071, 0,
		0.3536, -0.3536, 0, 0.5000, 0, 0, -0.7071, 0,
		0.3536, -0.3536, 0, -0.5000, 0, 0, 0, 0.7071,
		0.3536, -0.3536, 0, -0.5000, 0, 0, 0, -0.7071
	};
	float mascara_inv[LONG_MASCARA] = {
		0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536,
		0.3536, 0.3536, 0.3536, 0.3536, -0.3536, -0.3536, -0.3536, -0.3536,
		0.5000, 0.5000, -0.5000, -0.5000, 0, 0, 0, 0,
		0, 0, 0, 0, 0.5000, 0.5000, -0.5000, -0.5000,
		0.7071, -0.7071, 0, 0, 0, 0, 0, 0,
		0, 0, 0.7071, -0.7071, 0, 0, 0, 0,
		0, 0, 0, 0, 0.7071, -0.7071, 0, 0,
		0, 0, 0, 0, 0, 0, 0.7071, -0.7071
	};

	// Step 3: Create Buffers for Image Data
	float* imagenOriginal = (float*)imagen.ptr<float>(0);
	float* imagenTransf1 = new float[numPixeles];
	float* imagenTransf2 = new float[numPixeles];
	float* imagenTransf3 = new float[numPixeles];
	float* imagenTransf4 = new float[numPixeles];

	clock_t timer1 = clock();
	setQuantization = true;

	// Step 4: Haar Wavelet Transform in Loop
	int newRow = filas, newCol = columnas, type = imagen.type();

	while (newRow > TARGET_SIZE && newCol > TARGET_SIZE) {
		// Call the OpenCL functions for MultDerecha and MultIzquierda
		executeMultDerechaOpenCL(context, queue, program, imagenOriginal, imagenTransf1, mascara_inv, newRow, newCol, setQuantization);
		executeMultIzquierdaOpenCL(context, queue, program, imagenTransf1, imagenTransf2, mascara, newRow, newCol, setQuantization);

		newRow /= 2;
		newCol /= 2;
		imagenOriginal = imagenTransf2;
	}
	executeMultDerechaOpenCL(context, queue, program, imagenTransf2, imagenTransf3, mascara_inv, newRow, newCol, setQuantization);
	executeMultIzquierdaOpenCL(context, queue, program, imagenTransf3, imagenTransf4, mascara, newRow, newCol, setQuantization);

	// Step 5: Further Steps - Zigzag, RLE, Huffman (either on CPU or OpenCL)
	Mat imagenWaveletTransformadaOpenCL(newRow, newCol, CV_32F, imagenTransf4);
	Mat ImageArray = toZigZag(imagenWaveletTransformadaOpenCL);
	Mat ImageArrayCompressed = encodeRLE(ImageArray);
	auto [root, huffmanCode] = huffmanEncode(ImageArrayCompressed);
	string huffmanEncodedString = encodeWithHuffman(ImageArrayCompressed, huffmanCode);
	timer1 = clock() - timer1;

	cout << "\n\nCompressing with OpenCL" << endl;
	string path = "../../Compressed_Image/compressedDataOpenCL.bin";
	saveCompressedImage(path, newRow, newCol, imagen.type(), huffmanEncodedString);

	int InputBitcost = imagen.rows * imagen.cols * 8;
	cout << "InputBitcost: " << InputBitcost << endl;
	float OutputBitcost = huffmanEncodedString.length();
	cout << "OutputBitcost: " << OutputBitcost << endl;
	uintmax_t originalFileSize = getFileSize(nombreImagen);
	cout << "Original image file size: " << originalFileSize << " bytes" << endl;
	uintmax_t compressedFileSize = filesystem::file_size(path);
	cout << "Compressed file size: " << compressedFileSize << " bytes" << endl;
	cout << "Compression Ratio: " << originalFileSize / (double)compressedFileSize << endl;

	String binaryString;
	loadCompressedImage(path, newRow, newCol, type, binaryString);
	// Decode the Huffman-encoded string using the root node
	ImageArray = decodeHuffman(binaryString, root, ImageArrayCompressed.cols);

	ImageArray = decodeRLE(ImageArrayCompressed);
	imagenWaveletTransformadaOpenCL = invZigZag(ImageArray);
	float* tempTransf = (float*)imagenWaveletTransformadaOpenCL.ptr<float>(0);

	setQuantization = false;
	while (newRow < filas && newCol < columnas) {
		// Apply the Haar wavelet transform
		executeMultDerechaOpenCL(context, queue, program, tempTransf, imagenTransf1, mascara, newRow, newCol, setQuantization);
		executeMultIzquierdaOpenCL(context, queue, program, imagenTransf1, imagenTransf2, mascara_inv, newRow, newCol, setQuantization);

		newRow *= 2;
		newCol *= 2;

		// Update imagenOriginal for the next iteration
		tempTransf = imagenTransf2;
	}
	executeMultDerechaOpenCL(context, queue, program, imagenTransf2, imagenTransf3, mascara, newRow, newCol, setQuantization);
	executeMultIzquierdaOpenCL(context, queue, program, imagenTransf3, imagenTransf4, mascara_inv, newRow, newCol, setQuantization);

	printf("Image Size: [%d, %d]\n", filas, columnas);
	printf("CPU Execution Time is %10.3f ms.\n", ((timer1) / double(CLOCKS_PER_SEC) * 1000));

	Mat imagenRecuperada(filas, columnas, CV_32F, imagenTransf4);

	if (filas > 1024 && columnas > 1024)
	{
		double scale = 1024 / (double)filas;
		// Resize the image to fit the window
		resize(imagen, imagen, Size(), scale, scale, INTER_LINEAR);
		resize(imagenRecuperada, imagenRecuperada, Size(), scale, scale, INTER_LINEAR);
	}

	imagenRecuperada.convertTo(imagenRecuperada, CV_64F);
	imagen.convertTo(imagen, CV_64F);

	// ------ IMAGE METRICS -----
	cout << "------IMAGE METRICS-----" << endl;
	// CR
	cout << "The CR value is: " << InputBitcost / OutputBitcost << endl;
	// MSE
	cout << "The MSE value is: " << qm::eqm(imagen, imagenRecuperada) << endl;
	// PSNR
	cout << "The PSNR value is: " << qm::psnr(imagen, imagenRecuperada, 1) << endl;
	// SSIM
	//cout << "The SSIM value is: " << qm::ssim(imagen, imagenRecuperada, 1) << endl;

	imagenRecuperada.convertTo(imagenRecuperada, CV_8UC1);
	imagen.convertTo(imagen, CV_8UC1);

	imshow("Original", imagen);
	imshow("Compressed on OpenCL", imagenRecuperada);

	delete[] imagenTransf1;
	delete[] imagenTransf2;
	delete[] imagenTransf3;
	delete[] imagenTransf4;

	// Clean up OpenCL
	clReleaseDevice(device);
	clReleaseContext(context);
	clReleaseCommandQueue(queue);
	clReleaseProgram(program);

	return timer1;
}

double mainOMP(const string nombreImagen) {
	Mat imagen;
	bool setQuantization;
	float* imagenOriginal;
	imagen = imread(nombreImagen.c_str(), IMREAD_GRAYSCALE);
	imagen.convertTo(imagen, CV_32F);
	if (imagen.empty()) {
		cout << "Empty Image...";
	}
	int filas = imagen.rows;
	int columnas = imagen.cols;
	const size_t numPixeles = filas * columnas;
	float mascara[LONG_MASCARA] = {
		0.3536, 0.3536, 0.5000, 0, 0.7071, 0, 0, 0,
		0.3536, 0.3536, 0.5000, 0, -0.7071, 0, 0, 0,
		0.3536, 0.3536, -0.5000, 0, 0, 0.7071, 0, 0,
		0.3536, 0.3536, -0.5000, 0, 0, -0.7071, 0, 0,
		0.3536, -0.3536, 0, 0.5000, 0, 0, 0.7071, 0,
		0.3536, -0.3536, 0, 0.5000, 0, 0, -0.7071, 0,
		0.3536, -0.3536, 0, -0.5000, 0, 0, 0, 0.7071,
		0.3536, -0.3536, 0, -0.5000, 0, 0, 0, -0.7071
	};
	float mascara_inv[LONG_MASCARA] = {
		0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536,
		0.3536, 0.3536, 0.3536, 0.3536, -0.3536, -0.3536, -0.3536, -0.3536,
		0.5000, 0.5000, -0.5000, -0.5000, 0, 0, 0, 0,
		0, 0, 0, 0, 0.5000, 0.5000, -0.5000, -0.5000,
		0.7071, -0.7071, 0, 0, 0, 0, 0, 0,
		0, 0, 0.7071, -0.7071, 0, 0, 0, 0,
		0, 0, 0, 0, 0.7071, -0.7071, 0, 0,
		0, 0, 0, 0, 0, 0, 0.7071, -0.7071
	};
	imagenOriginal = (float*)imagen.ptr<float>(0);
	float* imagenTransf1 = new float[numPixeles];
	float* imagenTransf2 = new float[numPixeles];
	float* imagenTransf3 = new float[numPixeles];
	float* imagenTransf4 = new float[numPixeles];

	int newRow = filas;
	int newCol = columnas;
	int type = imagen.type();

	clock_t timer1 = clock();
	setQuantization = true;

	while (newRow > TARGET_SIZE && newCol > TARGET_SIZE) {
		// Apply the Haar wavelet transform
		MultDerechaCPU_OpenMP(imagenOriginal, imagenTransf1, mascara_inv, newRow, newCol, setQuantization);
		MultIzquierdaCPU_OpenMP(imagenTransf1, imagenTransf2, mascara, newRow, newCol, setQuantization);

		// Halve the rows and columns for the next iteration
		newRow /= 2;
		newCol /= 2;

		// Update imagenOriginal for the next iteration
		imagenOriginal = imagenTransf2;
	}
	MultDerechaCPU_OpenMP(imagenTransf2, imagenTransf3, mascara_inv, newRow, newCol, setQuantization);
	MultIzquierdaCPU_OpenMP(imagenTransf3, imagenTransf4, mascara, newRow, newCol, setQuantization);

	Mat imagenWaveletTransformadaCPU(newRow, newCol, CV_32F, imagenTransf4);
	Mat ImageArray = toZigZag(imagenWaveletTransformadaCPU);
	Mat ImageArrayCompressed = encodeRLE(ImageArray);
	// Get the Huffman tree and the codes
	auto [root, huffmanCode] = huffmanEncode(ImageArrayCompressed);

	// Encode the image data using the Huffman codes
	string huffmanEncodedString = encodeWithHuffman(ImageArrayCompressed, huffmanCode);

	timer1 = clock() - timer1;

	cout << "\n\nCompressing with CPU & OpenMP" << endl;
	string path = "../../Compressed_Image/compressedDataOMP.bin";
	saveCompressedImage(path, filas, columnas, imagen.type(), huffmanEncodedString);

	int InputBitcost = imagen.rows * imagen.cols * 8;
	cout << "InputBitcost: " << InputBitcost << endl;
	float OutputBitcost = huffmanEncodedString.length();
	cout << "OutputBitcost: " << OutputBitcost << endl;
	uintmax_t originalFileSize = getFileSize(nombreImagen);
	cout << "Original image file size: " << originalFileSize << " bytes" << endl;
	uintmax_t compressedFileSize = filesystem::file_size(path);
	cout << "Compressed file size: " << compressedFileSize << " bytes" << endl;
	cout << "Compression Ratio: " << originalFileSize / (double)compressedFileSize << endl;

	String binaryString;
	loadCompressedImage(path, filas, columnas, type, binaryString);

	// Decode the Huffman-encoded string using the root node
	ImageArray = decodeHuffman(binaryString, root, ImageArrayCompressed.cols);
	ImageArray = decodeRLE(ImageArrayCompressed);
	imagenWaveletTransformadaCPU = invZigZag(ImageArray);

	setQuantization = false;

	float* tempTransf = (float*)imagenWaveletTransformadaCPU.ptr<float>(0);

	while (newRow < filas && newCol < columnas) {
		// Apply the Haar wavelet transform
		MultDerechaCPU_OpenMP(tempTransf, imagenTransf1, mascara, newRow, newCol, setQuantization);
		MultIzquierdaCPU_OpenMP(imagenTransf1, imagenTransf2, mascara_inv, newRow, newCol, setQuantization);

		// Halve the rows and columns for the next iteration
		newRow *= 2;
		newCol *= 2;

		// Update imagenOriginal for the next iteration
		tempTransf = imagenTransf2;
	}
	MultDerechaCPU_OpenMP(imagenTransf2, imagenTransf3, mascara, newRow, newCol, setQuantization);
	MultIzquierdaCPU_OpenMP(imagenTransf3, imagenTransf4, mascara_inv, newRow, newCol, setQuantization);

	//timer1 = clock() - timer1;

	printf("Image Size: [%d, %d]\n", filas, columnas);
	printf("CPU with OpenMP Execution Time is %10.3f ms.\n", ((timer1) / double(CLOCKS_PER_SEC) * 1000));

	Mat imagenRecuperada(filas, columnas, CV_32F, imagenTransf4);

	if (filas > 1024 && columnas > 1024)
	{
		double scale = 1024 / (double)filas;
		// Resize the image to fit the window
		resize(imagen, imagen, Size(), scale, scale, INTER_LINEAR);
		resize(imagenRecuperada, imagenRecuperada, Size(), scale, scale, INTER_LINEAR);
	}

	imagenRecuperada.convertTo(imagenRecuperada, CV_64F);
	imagen.convertTo(imagen, CV_64F);

	// ------ IMAGE METRICS -----
	cout << "------IMAGE METRICS-----" << endl;
	// CR
	cout << "The CR value is: " << InputBitcost / OutputBitcost << endl;
	// MSE
	cout << "The MSE value is: " << qm::eqm(imagen, imagenRecuperada) << endl;
	// PSNR
	cout << "The PSNR value is: " << qm::psnr(imagen, imagenRecuperada, 1) << endl;
	// SSIM
	//cout << "The SSIM value is: " << qm::ssim(imagen, imagenRecuperada, 1) << endl;

	imagenRecuperada.convertTo(imagenRecuperada, CV_8UC1);
	imagen.convertTo(imagen, CV_8UC1);

	imshow("Original", imagen);
	imshow("Compressed on CPU with OpenMP", imagenRecuperada);

	delete[] imagenTransf1;
	delete[] imagenTransf2;
	delete[] imagenTransf3;
	delete[] imagenTransf4;

	return timer1;
}

int main(int args, char** argvs) {
	// Set OpenCV logging level to silent
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

	string imagepath;
	cout << "Enter image path: ";
	cin >> imagepath;

	double ori, gpu, omp, mpi, opencl;
	ori = mainCPU(imagepath);
	gpu = mainGPU(imagepath);
	opencl = mainOpenCL(imagepath);
	omp = mainOMP(imagepath);

	cout << "\n\nThe performance gain from GPU is " << ori / gpu << " times faster." << endl;
	cout << "The performance gain from OpenCL is " << ori / opencl << " times faster." << endl;
	cout << "The performance gain from OpenMP is " << ori / omp << " times faster." << endl;

	vector<double> executionTimes = { ori, gpu, opencl, omp };
	vector<string> labels = { "CPU", "CUDA GPU", "OpenCL GPU", "OpenMP" };

	drawBarChart(executionTimes, labels);

	waitKey(0);
	return 0;
}