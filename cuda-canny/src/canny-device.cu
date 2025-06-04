#include <stdio.h>

// CLE 24'25

// Sobel operator for X and Y directions
__constant__ float SOBEL_X[9];
__constant__ float SOBEL_Y[9];

typedef int pixel_t;

#define MAX_BRIGHTNESS 255

__global__ void convolution_kernel(const pixel_t *in, pixel_t *out, const float *kernel, const int width, const int height, const int kernel_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int kernel_half_size = kernel_size / 2;

    // check if pixel valid
    if (x < kernel_half_size || x >= width - kernel_half_size || y < kernel_half_size || y >= height - kernel_half_size)
        return;

    float pixel = 0.0f;
    int index = 0;

    for (int j = -kernel_half_size; j <= kernel_half_size; j++) {
        for (int i = -kernel_half_size; i <= kernel_half_size; i++) {
            pixel += in[(y - j) * width + x - i] * kernel[index];
            index++;
        }
    }
    
    out[y * width + x] = pixel;
}

__global__ void min_max_kernel(const pixel_t *in, const int width, const int height, pixel_t *min, pixel_t *max)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int index = y * width + x;
    pixel_t pixel = in[index];

    atomicMin(min, pixel);
    atomicMax(max, pixel);
}

__global__ void normalize_kernel(pixel_t *d_inout, int nx, int ny, int kernel, const int *d_min, const int *d_max) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; 
    int y = blockIdx.y * blockDim.y + threadIdx.y; 

    int kernel_half_size = kernel / 2;

    if (x >= kernel_half_size && x < nx - kernel_half_size && y >= kernel_half_size && y < ny - kernel_half_size) {
        int index = y * nx + x;

        int min_val = *d_min;
        int max_val = *d_max;

        float normalized = MAX_BRIGHTNESS * ((float)d_inout[index] - (float)min_val) / ((float)(max_val - min_val));
        d_inout[index] = (pixel_t)normalized;
    }
}

void gaussian_filter_device(const pixel_t *in, pixel_t *out,
                     const int width, const int height, const float sigma,
                    dim3 block, dim3 grid)
{
    const int kernel_size = 2 * (int)(2 * sigma) + 3;
    const float mean = (float)floor(kernel_size / 2.0);
    float h_kernel[kernel_size * kernel_size];

    // fprintf(stderr, "gaussian_filter_device: kernel size %d, sigma=%g\n", kernel_size, sigma);
    size_t index = 0;
    for (int i = 0; i < kernel_size; i++)
        for (int j = 0; j < kernel_size; j++) {
            h_kernel[index] = exp(-0.5 * (pow((i - mean) / sigma, 2.0) + pow((j - mean) / sigma, 2.0))) / (2 * M_PI * sigma * sigma);
            index++;
        }

    float *d_kernel;
    cudaMalloc(&d_kernel, sizeof(float) * kernel_size * kernel_size);
    cudaMemcpy(d_kernel, h_kernel, sizeof(float) * kernel_size * kernel_size, cudaMemcpyHostToDevice);
    convolution_kernel<<<grid, block>>>(in, out, d_kernel, width, height, kernel_size);

    pixel_t *d_min, *d_max;
    cudaMalloc(&d_min, sizeof(pixel_t));
    cudaMalloc(&d_max, sizeof(pixel_t));

    min_max_kernel<<<grid, block>>>(out, width, height, d_min, d_max);
    normalize_kernel<<<grid, block>>>(out, width, height, kernel_size, d_min, d_max);
    
    cudaFree(d_kernel);
    cudaFree(d_min);
    cudaFree(d_max);
}

__global__ void merge_gradients_kernel(const pixel_t *gx, const pixel_t *gy, pixel_t *out, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; 
    int y = blockIdx.y * blockDim.y + threadIdx.y; 
    
    if (x >= width || y >= height) return;

    int idx = y * width + x;

    out[idx] = hypotf(gx[idx], gy[idx]);
}

__global__ void non_maximum_suppression_kernel(const pixel_t *d_gradient_x, const pixel_t *d_gradient_y, const pixel_t *gradient_mag, int *nms,
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

    const float dir = (float)(fmod(atan2f(d_gradient_y[index],d_gradient_x[index]) + M_PI,M_PI) / M_PI) * 8;
    if (((dir <= 1 || dir > 7) && gradient_mag[index] > gradient_mag[ee] &&
                gradient_mag[index] > gradient_mag[ww]) || // 0 deg
                ((dir > 1 && dir <= 3) && gradient_mag[index] > gradient_mag[nw] &&
                gradient_mag[index] > gradient_mag[se]) || // 45 deg
                ((dir > 3 && dir <= 5) && gradient_mag[index] > gradient_mag[nn] &&
                gradient_mag[index] > gradient_mag[ss]) || // 90 deg
                ((dir > 5 && dir <= 7) && gradient_mag[index] > gradient_mag[ne] &&
                gradient_mag[index] > gradient_mag[sw]))   // 135 deg
                nms[index] = gradient_mag[index];
            else
                nms[index] = 0;
}


__global__ void first_edges_kernel(const int *nms, int *edges, int width, int height, int tmax)
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

__global__ void hysteresis_edges_kernel(const int *nms, int *edges, int width, int height, int tmin, bool *d_changed)
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

void cannyDevice(const int *h_idata, const int width, const int height, const int tmin, const int tmax, const float sigma, int *h_odata, int block_size)
{
    const int size = width * height;
        
    dim3 block(block_size, block_size);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    pixel_t *d_idata, *d_odata, *d_gradient_x, *d_gradient_y, *d_nms;
    cudaMalloc(&d_idata, sizeof(pixel_t) * size);
    cudaMalloc(&d_odata, sizeof(pixel_t) * size);
    cudaMalloc(&d_gradient_x, sizeof(pixel_t) * size);
    cudaMalloc(&d_gradient_y, sizeof(pixel_t) * size);
    cudaMalloc(&d_nms, sizeof(pixel_t) * size);

    cudaMemcpy(d_idata, h_idata, sizeof(pixel_t) * size, cudaMemcpyHostToDevice);

    // Gaussian filter
    gaussian_filter_device(d_idata, d_odata, width, height, sigma, block, grid);

    // Gradients
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

    // Gradient along x
    convolution_kernel<<<grid, block>>>(d_odata, d_gradient_x, d_sobel_x, width, height, 3);
    // Gradient along y
    convolution_kernel<<<grid, block>>>(d_odata, d_gradient_y, d_sobel_y, width, height, 3);
    // Merging gradients
    merge_gradients_kernel<<<grid, block>>>(d_gradient_x, d_gradient_y, d_odata, width, height);
    
    // Non-maximum suppression, straightforward implementation.
    non_maximum_suppression_kernel<<<grid, block>>>(d_gradient_x, d_gradient_y, d_odata, d_nms, width, height);
    first_edges_kernel<<<grid, block>>>(d_nms, d_odata, width, height, tmax);
    
    // edges with nms >= tmax
    bool h_changed;
    bool *d_changed;
    cudaMalloc(&d_changed, sizeof(bool));

    // edges with nms >= tmin && neighbor is edge
    do {
        h_changed = false;
        cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice);

        hysteresis_edges_kernel<<<grid, block>>>(d_nms, d_odata, width, height, tmin, d_changed);
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
