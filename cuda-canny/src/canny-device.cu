
// CLE 24'25

// Sobel operator for X and Y directions
__constant__ float SOBEL_X[9];
__constant__ float SOBEL_Y[9];

__global__ void convolutionKernel(const float *in, float *out, const float *kernel,
                                  const int nx, const int ny, const int kn)
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;

    const int khalf = kn / 2;

    // Only compute for valid output pixels (inside border)
    if (m < khalf || m >= nx - khalf || n < khalf || n >= ny - khalf)
        return;

    float pixel = 0.0f;
    int c = 0;

    for (int j = -khalf; j <= khalf; j++) {
        for (int i = -khalf; i <= khalf; i++) {
            int xi = m - i;
            int yj = n - j;
            pixel += in[yj * nx + xi] * kernel[c];
            c++;
        }
    }

    out[n * nx + m] = pixel;
}

__global__ void mergeGradientsKernel(const float *gx, const float *gy, float *G, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    G[idx] = hypotf(gx[idx], gy[idx]);
}

__global__ void non_maximum_suppressionKernel(const float *Gx, const float *Gy, const float *G, int *nms,
                                              int nx, int ny)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || y < 1 || x >= nx - 1 || y >= ny - 1) return;

    int c = y * nx + x;

    float angle = atan2f(Gy[c], Gx[c]);
    // Normalize angle to [0, PI)
    angle = fmodf(angle + M_PI, M_PI);

    // Convert angle to 8 sectors (0 to 7)
    int dir = (int)(angle / M_PI * 8) % 8;

    bool keep = false;
    switch (dir) {
        case 0: // 0 degrees (horizontal)
        case 7:
            keep = (G[c] > G[c - 1] && G[c] > G[c + 1]);
            break;
        case 1: // 45 degrees
        case 2:
            keep = (G[c] > G[c - nx + 1] && G[c] > G[c + nx - 1]);
            break;
        case 3: // 90 degrees (vertical)
        case 4:
            keep = (G[c] > G[c - nx] && G[c] > G[c + nx]);
            break;
        case 5: // 135 degrees
        case 6:
            keep = (G[c] > G[c - nx - 1] && G[c] > G[c + nx + 1]);
            break;
    }

    nms[c] = keep ? (int)G[c] : 0;
}

#define MAX_BRIGHTNESS 255

__global__ void first_edgesKernel(const int *nms, int *edges,
                                  int nx, int ny, int tmax)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || y < 1 || x >= nx - 1 || y >= ny - 1) return;

    int c = y * nx + x;
    if (nms[c] >= tmax)
        edges[c] = MAX_BRIGHTNESS;
    else
        edges[c] = 0;
}

__global__ void hysteresis_edgesKernel(const int *nms, int *edges,
                                       int nx, int ny, int tmin, bool *d_changed)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || y < 1 || x >= nx - 1 || y >= ny - 1) return;

    int c = y * nx + x;

    if (nms[c] >= tmin && edges[c] == 0) {
        // Check 8 neighbors for edge presence
        if (edges[c - nx] || edges[c + nx] || edges[c - 1] || edges[c + 1] ||
            edges[c - nx - 1] || edges[c - nx + 1] || edges[c + nx - 1] || edges[c + nx + 1]) {
            edges[c] = MAX_BRIGHTNESS;
            *d_changed = true;
        }
    }
}

void min_max(const float *in, const int nx, const int ny, float *min, float *max)
{
    *min = *max = in[0];
    for (int i = 1; i < nx * ny; i++) {
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


// canny edge detector code to run on the GPU
void cannyDevice( const int *h_idata, const int w, const int h,
                 const int tmin, const int tmax,
                 const float sigma,
                 int * h_odata)
{
    const int size = w * h;
    const int kn = 2 * (int)(2 * sigma) + 3;

    float *h_floatInput = new float[size];
    for (int i = 0; i < size; i++)
        h_floatInput[i] = static_cast<float>(h_idata[i]);

    float *d_input, *d_blur, *d_gx, *d_gy, *d_G;
    int *d_nms, *d_edges;
    cudaMalloc(&d_input, sizeof(float) * size);
    cudaMalloc(&d_blur, sizeof(float) * size);
    cudaMalloc(&d_gx, sizeof(float) * size);
    cudaMalloc(&d_gy, sizeof(float) * size);
    cudaMalloc(&d_G, sizeof(float) * size);
    cudaMalloc(&d_nms, sizeof(int) * size);
    cudaMalloc(&d_edges, sizeof(int) * size);

    cudaMemcpy(d_input, h_floatInput, sizeof(float) * size, cudaMemcpyHostToDevice);

    // Generate Gaussian kernel on host
    float *h_kernel = new float[kn * kn];
    const float mean = floor(kn / 2.0);
    size_t c = 0;
    for (int i = 0; i < kn; ++i) {
        for (int j = 0; j < kn; ++j) {
            h_kernel[c++] = expf(-0.5f * (powf((i - mean) / sigma, 2.0f) +
                                           powf((j - mean) / sigma, 2.0f))) /
                            (2.0f * M_PI * sigma * sigma);
        }
    }

    float *d_kernel;
    cudaMalloc(&d_kernel, sizeof(float) * kn * kn);
    cudaMemcpy(d_kernel, h_kernel, sizeof(float) * kn * kn, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((w + 15) / 16, (h + 15) / 16);

    // Apply Gaussian blur
    convolutionKernel<<<gridSize, blockSize>>>(d_input, d_blur, d_kernel, w, h, kn);

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
    convolutionKernel<<<gridSize, blockSize>>>(d_blur, d_gx, d_sobel_x, w, h, 3);
    convolutionKernel<<<gridSize, blockSize>>>(d_blur, d_gy, d_sobel_y, w, h, 3);

    // Merge gradients magnitude
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    mergeGradientsKernel<<<blocks, threads>>>(d_gx, d_gy, d_G, size);

    // Wait for merge to finish and copy to host to compute min/max
    cudaDeviceSynchronize();
    float *h_G = new float[size];
    cudaMemcpy(h_G, d_G, sizeof(float) * size, cudaMemcpyDeviceToHost);

    float minVal, maxVal;
    min_max(h_G, w, h, &minVal, &maxVal);

    // Normalize gradient magnitude to 0-255 and store in edges (temporary)
    normalizeKernel<<<blocks, threads>>>(d_G, d_nms, size, minVal, maxVal);

    cudaDeviceSynchronize();

    // Non-maximum suppression
    non_maximum_suppressionKernel<<<gridSize, blockSize>>>(d_gx, d_gy, d_G, d_nms, w, h);
    cudaDeviceSynchronize();

    // First edges thresholding
    first_edgesKernel<<<gridSize, blockSize>>>(d_nms, d_edges, w, h, tmax);
    cudaDeviceSynchronize();

    // Hysteresis thresholding loop
    bool h_changed;
    bool *d_changed;
    cudaMalloc(&d_changed, sizeof(bool));

    do {
        h_changed = false;
        cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice);

        hysteresis_edgesKernel<<<gridSize, blockSize>>>(d_nms, d_edges, w, h, tmin, d_changed);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
    } while (h_changed);

    cudaMemcpy(h_odata, d_edges, sizeof(int) * size, cudaMemcpyDeviceToHost);

    // Cleanup
    delete[] h_floatInput;
    delete[] h_kernel;
    delete[] h_G;

    cudaFree(d_input);
    cudaFree(d_blur);
    cudaFree(d_gx);
    cudaFree(d_gy);
    cudaFree(d_G);
    cudaFree(d_kernel);
    cudaFree(d_sobel_x);
    cudaFree(d_sobel_y);
    cudaFree(d_nms);
    cudaFree(d_edges);
    cudaFree(d_changed);
}
