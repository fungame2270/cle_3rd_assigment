#include <stdio.h>

// CLE 24'25

// Sobel operator for X and Y directions
__constant__ float SOBEL_X[9];
__constant__ float SOBEL_Y[9];

typedef int pixel_t;

#define MAX_BRIGHTNESS 255

__global__ void convolutionKernel(const pixel_t *in, pixel_t *out, const float *kernel,
                                  const int width, const int height, const int kernelSize)
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;

    const int kernelHalfSize = kernelSize / 2;

    // Only compute for valid output pixels (inside border)
    if (m < kernelHalfSize || m >= width - kernelHalfSize || n < kernelHalfSize || n >= height - kernelHalfSize)
        return;

    float pixel = 0.0f;
    int index = 0;

    for (int j = -kernelHalfSize; j <= kernelHalfSize; j++) {
        for (int i = -kernelHalfSize; i <= kernelHalfSize; i++) {
            int xi = m - i;
            int yj = n - j;
            pixel += in[yj * width + xi] * kernel[index];
            index++;
        }
    }

    out[n * width + m] = pixel;
}

void min_max_host(const pixel_t *in, const int width, const int height, pixel_t *min, pixel_t *max)
{
    *min = *max = in[0];
    for (int i = 1; i < width * height; i++) {
        if (in[i] < *min) *min = in[i];
        if (in[i] > *max) *max = in[i];
    }
}

void normalize_host(  pixel_t *inout,
               const int nx, const int ny, const int kn,
               const int min, const int max)
{
    const int khalf = kn / 2;

    for (int m = khalf; m < nx - khalf; m++)
        for (int n = khalf; n < ny - khalf; n++) {

            pixel_t pixel = MAX_BRIGHTNESS * ((int)inout[n * nx + m] -(float) min) / ((float)max - (float)min);
            inout[n * nx + m] = pixel;
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
            h_kernel[index] = exp(-0.5 * (pow((i - mean) / sigma, 2.0) +
                            pow((j - mean) / sigma, 2.0)))
                / (2 * M_PI * sigma * sigma);
            index++;
        }

    float *d_kernel;
    cudaMalloc(&d_kernel, sizeof(float) * kernel_size * kernel_size);
    cudaMemcpy(d_kernel, h_kernel, sizeof(float) * kernel_size * kernel_size, cudaMemcpyHostToDevice);

    dim3 block_size(16, 16);
    dim3 grid_size((width + 15) / 16, (height + 15) / 16);
    convolutionKernel<<<grid_size, block_size>>>(in, out, d_kernel, width, height, kernel_size);
    
    // run min max and normalize on host
    pixel_t *h_out = new pixel_t[width * height];
    cudaMemcpy(h_out, out, sizeof(pixel_t) * width * height, cudaMemcpyDeviceToHost);

    pixel_t min, max;
    min_max_host(h_out, width, height, &min, &max);
    normalize_host(h_out, width, height, kernel_size, min, max);
    
    cudaMemcpy(out, h_out, sizeof(pixel_t) * width * height, cudaMemcpyHostToDevice);
    delete[] h_out;
    cudaFree(d_kernel);
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

    if (x < 1 || y < 1 || x >= width - 1 || y >= height - 1) return;

    int index = y * width + x;

    float angle = atan2f(d_gradientY[index], d_gradientX[index]);
    // Normalize angle to [0, PI)
    angle = fmodf(angle + M_PI, M_PI);

    // Convert angle to 8 sectors (0 to 7)
    int dir = (int)(angle / M_PI * 8) % 8;

    bool keep = false;
    switch (dir) {
        case 0: // 0 degrees (horizontal)
        case 7:
            keep = (gradientMag[index] > gradientMag[index - 1] && gradientMag[index] > gradientMag[index + 1]);
            break;
        case 1: // 45 degrees
        case 2:
            keep = (gradientMag[index] > gradientMag[index - width + 1] && gradientMag[index] > gradientMag[index + width - 1]);
            break;
        case 3: // 90 degrees (vertical)
        case 4:
            keep = (gradientMag[index] > gradientMag[index - width] && gradientMag[index] > gradientMag[index + width]);
            break;
        case 5: // 135 degrees
        case 6:
            keep = (gradientMag[index] > gradientMag[index - width - 1] && gradientMag[index] > gradientMag[index + width + 1]);
            break;
    }

    nms[index] = keep ? (int)gradientMag[index] : 0;
}


__global__ void first_edgesKernel(const int *nms, int *edges,
                                  int width, int height, int tmax)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || y < 1 || x >= width - 1 || y >= height - 1) return;

    int index = y * width + x;
    if (nms[index] >= tmax)
        edges[index] = MAX_BRIGHTNESS;
    else
        edges[index] = 0;
}

__global__ void hysteresis_edgesKernel(const int *nms, int *edges,
                                       int width, int height, int tmin, bool *d_changed)
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

void cannyDevice( const int *h_idata, 
                 const int width, const int height,
                 const int tmin, const int tmax,
                 const float sigma,
                 int *h_odata)
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

    cudaDeviceSynchronize();

    non_maximum_suppressionKernel<<<grid_size, block_size>>>(d_gradient_x, d_gradient_y, d_odata, d_nms, width, height);
    
    cudaDeviceSynchronize();

    first_edgesKernel<<<grid_size, block_size>>>(d_nms, d_odata, width, height, tmax);
    
    cudaDeviceSynchronize();

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
