#include <stdio.h>

// CLE 24'25

// Sobel operator for X and Y directions
__constant__ float SOBEL_X[9];
__constant__ float SOBEL_Y[9];

typedef int pixel_t;

#define MAX_BRIGHTNESS 255

__global__ void convolutionKernel(const pixel_t *in, pixel_t *out, const float *kernel, const int width, const int height, const int kernelSize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int kernelHalfSize = kernelSize / 2;

    // check if pixel valid
    if (x < kernelHalfSize || x >= width - kernelHalfSize || y < kernelHalfSize || y >= height - kernelHalfSize)
        return;

    float pixel = 0.0f;
    int index = 0;

    for (int j = -kernelHalfSize; j <= kernelHalfSize; j++) {
        for (int i = -kernelHalfSize; i <= kernelHalfSize; i++) {
            pixel += in[(y - j) * width + x - i] * kernel[index];
            index++;
        }
    }
    
    out[y * width + x] = pixel;
}

__global__ void min_max_device(const pixel_t *in, const int width, const int height, pixel_t *min, pixel_t *max)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int index = y * width + x;
    pixel_t pixel = in[index];

    atomicMin(min, pixel);
    atomicMax(max, pixel);
}

__global__ void normalize_device(pixel_t *d_inout, int nx, int ny, int kernel, const int *d_min, const int *d_max) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; 
    int y = blockIdx.y * blockDim.y + threadIdx.y; 

    int kernelHalfSize = kernel / 2;

    if (x >= kernelHalfSize && x < nx - kernelHalfSize && y >= kernelHalfSize && y < ny - kernelHalfSize) {
        int index = y * nx + x;

        int min_val = *d_min;
        int max_val = *d_max;

        float normalized = MAX_BRIGHTNESS * ((float)d_inout[index] - (float)min_val) / ((float)(max_val - min_val));
        d_inout[index] = (pixel_t)normalized;
    }
}

void gaussian_filter_device(const pixel_t *in, pixel_t *out,
                     const int width, const int height, const float sigma)
{
    const int kernel_size = 2 * (int)(2 * sigma) + 3;
    const float mean = (float)floor(kernel_size / 2.0);
    float h_kernel[kernel_size * kernel_size];

    fprintf(stderr, "gaussian_filter_device: kernel size %d, sigma=%g\n", kernel_size, sigma);
    size_t index = 0;
    for (int i = 0; i < kernel_size; i++)
        for (int j = 0; j < kernel_size; j++) {
            h_kernel[index] = exp(-0.5 * (pow((i - mean) / sigma, 2.0) + pow((j - mean) / sigma, 2.0))) / (2 * M_PI * sigma * sigma);
            index++;
        }

    float *d_kernel;
    cudaMalloc(&d_kernel, sizeof(float) * kernel_size * kernel_size);
    cudaMemcpy(d_kernel, h_kernel, sizeof(float) * kernel_size * kernel_size, cudaMemcpyHostToDevice);

    dim3 block_size(16, 16);
    dim3 grid_size((width + 15) / 16, (height + 15) / 16);
    convolutionKernel<<<grid_size, block_size>>>(in, out, d_kernel, width, height, kernel_size);

    pixel_t *d_min, *d_max;
    cudaMalloc(&d_min, sizeof(pixel_t));
    cudaMalloc(&d_max, sizeof(pixel_t));

    min_max_device<<<grid_size, block_size>>>(out, width, height, d_min, d_max);
    normalize_device<<<grid_size, block_size>>>(out, width, height, kernel_size, d_min, d_max);
    
    cudaFree(d_kernel);
    cudaFree(d_min);
    cudaFree(d_max);
}

__global__ void mergeGradientsKernel(const pixel_t *gx, const pixel_t *gy, pixel_t *out, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    out[idx] = hypotf(gx[idx], gy[idx]);
}

__global__ void non_maximum_suppressionKernel(const pixel_t *d_gradientX, const pixel_t *d_gradientY, const pixel_t *gradientMag, int *nms,
                                              int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    //skipts border
    if (x < 1 || y < 1 || x >= width - 1 || y >= height - 1) return;

    int index = y * width + x;

    const int nn = index - width;
    const int ss = index + width;
    const int ww = index + 1;
    const int ee = index - 1;
    const int nw = nn + 1;
    const int ne = nn - 1;
    const int sw = ss + 1;
    const int se = ss - 1;

    const float dir = (float)(fmod(atan2f(d_gradientY[index],d_gradientX[index]) + M_PI,M_PI) / M_PI) * 8;
    if (((dir <= 1 || dir > 7) && gradientMag[index] > gradientMag[ee] &&
                gradientMag[index] > gradientMag[ww]) || // 0 deg
                ((dir > 1 && dir <= 3) && gradientMag[index] > gradientMag[nw] &&
                gradientMag[index] > gradientMag[se]) || // 45 deg
                ((dir > 3 && dir <= 5) && gradientMag[index] > gradientMag[nn] &&
                gradientMag[index] > gradientMag[ss]) || // 90 deg
                ((dir > 5 && dir <= 7) && gradientMag[index] > gradientMag[ne] &&
                gradientMag[index] > gradientMag[sw]))   // 135 deg
                nms[index] = gradientMag[index];
            else
                nms[index] = 0;
}


__global__ void first_edgesKernel(const int *nms, int *edges, int width, int height, int tmax)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // check position valid
    if (x < 1 || y < 1 || x >= width - 1 || y >= height - 1) return;

    int index = y * width + x;
    if (nms[index] >= tmax)
        edges[index] = MAX_BRIGHTNESS;
    else
        edges[index] = 0;
}

__global__ void hysteresis_edgesKernel(const int *nms, int *edges, int width, int height, int tmin, bool *d_changed)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || y < 1 || x >= width - 1 || y >= height - 1) return;

    int index = y * width + x;

    if (nms[index] >= tmin && edges[index] == 0) {
        // Check 8 neighbors for edge presence
        if (edges[index - width] || edges[index + width] || edges[index - 1] || edges[index + 1] ||
            edges[index - width - 1] || edges[index - width + 1] || edges[index + width - 1] || edges[index + width + 1]) {
                edges[index] = MAX_BRIGHTNESS;
                *d_changed = true;
        }
    }
}

void cannyDevice(const int *h_idata, const int width, const int height, const int tmin, const int tmax, const float sigma, int *h_odata)
{
    const int size = width * height;

    pixel_t *d_idata, *d_odata, *d_gradient_x, *d_gradient_y, *d_nms;
    cudaMalloc(&d_idata, sizeof(pixel_t) * size);
    cudaMalloc(&d_odata, sizeof(pixel_t) * size);
    cudaMalloc(&d_gradient_x, sizeof(pixel_t) * size);
    cudaMalloc(&d_gradient_y, sizeof(pixel_t) * size);
    cudaMalloc(&d_nms, sizeof(pixel_t) * size);

    cudaMemcpy(d_idata, h_idata, sizeof(pixel_t) * size, cudaMemcpyHostToDevice);

    gaussian_filter_device(d_idata, d_odata, width, height, sigma);

    float h_sobel_x[9] = {-1, 0, 1,
                          -2, 0, 2,
                          -1, 0, 1};
    float h_sobel_y[9] = {1, 2, 1,
                          0, 0, 0,
                         -1, -2, -1};

    float *d_sobel_x, *d_sobel_y;
    cudaMalloc(&d_sobel_x, sizeof(float) * 9);
    cudaMalloc(&d_sobel_y, sizeof(float) * 9);
    cudaMemcpy(d_sobel_x, h_sobel_x, sizeof(float) * 9, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sobel_y, h_sobel_y, sizeof(float) * 9, cudaMemcpyHostToDevice);

    dim3 block_size(16, 16);
    dim3 grid_size((width + 15) / 16, (height + 15) / 16);

    convolutionKernel<<<grid_size, block_size>>>(d_odata, d_gradient_x, d_sobel_x, width, height, 3);
    convolutionKernel<<<grid_size, block_size>>>(d_odata, d_gradient_y, d_sobel_y, width, height, 3);

    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    mergeGradientsKernel<<<blocks, threads>>>(d_gradient_x, d_gradient_y, d_odata, size);
    non_maximum_suppressionKernel<<<grid_size, block_size>>>(d_gradient_x, d_gradient_y, d_odata, d_nms, width, height);
    first_edgesKernel<<<grid_size, block_size>>>(d_nms, d_odata, width, height, tmax);
    
    bool h_changed;
    bool *d_changed;
    cudaMalloc(&d_changed, sizeof(bool));

    do {
        h_changed = false;
        cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice);

        hysteresis_edgesKernel<<<grid_size, block_size>>>(d_nms, d_odata, width, height, tmin, d_changed);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
    } while (h_changed);

    cudaMemcpy(h_odata, d_odata, sizeof(pixel_t) * size, cudaMemcpyDeviceToHost);

    cudaFree(d_idata);
    cudaFree(d_odata);
    cudaFree(d_gradient_x);
    cudaFree(d_gradient_y);
    cudaFree(d_nms);
}
