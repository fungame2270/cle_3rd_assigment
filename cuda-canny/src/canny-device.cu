
// CLE 24'25

// Sobel operator for X and Y directions
__constant__ float SOBEL_X[9];
__constant__ float SOBEL_Y[9];

__global__ void convolutionKernel(const float *in, float *out, const float *kernel,
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

__global__ void mergeGradientsKernel(const float *gx, const float *gy, float *gradientMag, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    gradientMag[idx] = hypotf(gx[idx], gy[idx]);
}

__global__ void non_maximum_suppressionKernel(const float *d_gradientX, const float *d_gradientY, const float *gradientMag, int *nms,
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

#define MAX_BRIGHTNESS 255

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

void min_max(const float *in, const int width, const int height, float *min, float *max)
{
    *min = *max = in[0];
    for (int i = 1; i < width * height; i++) {
        if (in[i] < *min) *min = in[i];
        if (in[i] > *max) *max = in[i];
    }
}


__global__ void normalizeKernel(float *data, int *out, int size, float minVal, float maxVal)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    out[idx] = (int)(MAX_BRIGHTNESS * (data[idx] - minVal) / (maxVal - minVal));
}


void cannyDevice( const int *h_idata, const int width, const int height,
                 const int tmin, const int tmax,
                 const float sigma,
                 int * h_odata)
{
    const int size = width * height;
    const int kernelSize = 2 * (int)(2 * sigma) + 3;

    float *h_floatInput = new float[size];
    for (int i = 0; i < size; i++)
        h_floatInput[i] = static_cast<float>(h_idata[i]);

    float *d_inputImage, *d_blur, *d_gradientX, *d_gradientY, *d_gradientMag;
    int *d_nmsOutput, *d_edges;
    cudaMalloc(&d_inputImage, sizeof(float) * size);
    cudaMalloc(&d_blur, sizeof(float) * size);
    cudaMalloc(&d_gradientX, sizeof(float) * size);
    cudaMalloc(&d_gradientY, sizeof(float) * size);
    cudaMalloc(&d_gradientMag, sizeof(float) * size);
    cudaMalloc(&d_nmsOutput, sizeof(int) * size);
    cudaMalloc(&d_edges, sizeof(int) * size);

    cudaMemcpy(d_inputImage, h_floatInput, sizeof(float) * size, cudaMemcpyHostToDevice);

    // Generate Gaussian kernel on host
    float *h_gaussianKernel = new float[kernelSize * kernelSize];
    const float mean = floor(kernelSize / 2.0);
    size_t index = 0;
    for (int i = 0; i < kernelSize; ++i) {
        for (int j = 0; j < kernelSize; ++j) {
            h_gaussianKernel[index++] = expf(-0.5f * (powf((i - mean) / sigma, 2.0f) +
                                           powf((j - mean) / sigma, 2.0f))) /
                            (2.0f * M_PI * sigma * sigma);
        }
    }

    float *d_kernel;
    cudaMalloc(&d_kernel, sizeof(float) * kernelSize * kernelSize);
    cudaMemcpy(d_kernel, h_gaussianKernel, sizeof(float) * kernelSize * kernelSize, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15) / 16, (height + 15) / 16);

    // Apply Gaussian blur
    convolutionKernel<<<gridSize, blockSize>>>(d_inputImage, d_blur, d_kernel, width, height, kernelSize);

    // Sobel kernels
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

    // Compute gradients
    convolutionKernel<<<gridSize, blockSize>>>(d_blur, d_gradientX, d_sobel_x, width, height, 3);
    convolutionKernel<<<gridSize, blockSize>>>(d_blur, d_gradientY, d_sobel_y, width, height, 3);

    // Merge gradients magnitude
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    mergeGradientsKernel<<<blocks, threads>>>(d_gradientX, d_gradientY, d_gradientMag, size);

    // Wait for merge to finish and copy to host to compute min/max
    cudaDeviceSynchronize();
    float *h_gradientMag = new float[size];
    cudaMemcpy(h_gradientMag, d_gradientMag, sizeof(float) * size, cudaMemcpyDeviceToHost);

    float minVal, maxVal;
    min_max(h_gradientMag, width, height, &minVal, &maxVal);

    // Normalize gradient magnitude to 0-255 and store in edges (temporary)
    normalizeKernel<<<blocks, threads>>>(d_gradientMag, d_nmsOutput, size, minVal, maxVal);

    cudaDeviceSynchronize();

    // Non-maximum suppression
    non_maximum_suppressionKernel<<<gridSize, blockSize>>>(d_gradientX, d_gradientY, d_gradientMag, d_nmsOutput, width, height);
    cudaDeviceSynchronize();

    // First edges thresholding
    first_edgesKernel<<<gridSize, blockSize>>>(d_nmsOutput, d_edges, width, height, tmax);
    cudaDeviceSynchronize();

    // Hysteresis thresholding loop
    bool h_changed;
    bool *d_changed;
    cudaMalloc(&d_changed, sizeof(bool));

    do {
        h_changed = false;
        cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice);

        hysteresis_edgesKernel<<<gridSize, blockSize>>>(d_nmsOutput, d_edges, width, height, tmin, d_changed);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
    } while (h_changed);

    cudaMemcpy(h_odata, d_edges, sizeof(int) * size, cudaMemcpyDeviceToHost);

    // Cleanup
    delete[] h_floatInput;
    delete[] h_gaussianKernel;
    delete[] h_gradientMag;

    cudaFree(d_inputImage);
    cudaFree(d_blur);
    cudaFree(d_gradientX);
    cudaFree(d_gradientY);
    cudaFree(d_gradientMag);
    cudaFree(d_kernel);
    cudaFree(d_sobel_x);
    cudaFree(d_sobel_y);
    cudaFree(d_nmsOutput);
    cudaFree(d_edges);
    cudaFree(d_changed);
}
