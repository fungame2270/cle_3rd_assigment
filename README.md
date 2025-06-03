# CLE Third Assigment
## Implementation
### Convolution
The **convolutionKernel** performs 2D convolution in an image that is passed to the function as a one dimensional array. The function starts by calculating the image coordinates for the thread (x,y) in each thread and checks if the pixel is inside the boundaries, meaning it excludes borders based on kernel size. Then it iterates over the kernel that is centered in the pixel (x,y) and multiplies each value to the kernel weight at that window position and adds the result to an accumulator. Finally it stores the value in the output image array.
### Non Maximum Suppression
The **non_maximum_suppressionKernel** performs non-maximum suppression on an edge magnitude map. Like in the other function, each thread starts by calculating the pixel (x,y) that they will work on, skipping border pixels to avoid out-of-bounds access. Then it calculates the gradient direction using the arctangent of the X and Y gradient components and normalizes to one of the four directions 0º,45º,90º and 135º. After that it compares the pixel gradient to the neighbors in the direction. If the pixel has a higher magnitude compared to the neighbors then the nms of the pixel is kept has the magnitude else its set to zero, Preserving only the local maxima.
### First edges
The **first_edgesKernel** performs the first thresholding in the algorithm. Starts by getting the (x,y) values of the pixel and checking if the pixel is not on the border. It checks if non-maximum suppressed value at that pixel is greater than or equal to the threshold given **tmax**. If so the value in the edges array is set to **MAX_BRIGHTNESS**; otherwise, it is set to 0. 
### Hysteresis Edges
The **hysteresis_edgesKernel** is used in the hysteresis stage of Canny edge detection to connect weak edges to strong ones. It processes each pixel similarly to the other functions. Then, it checks if the value in the NMS array is greater than or equal to threshold **tmin** and if the pixel is not already marked as an edge, the kernel will check the 8 neighboring pixels in the edges array. If any neighbor is a strong edge (**MAX_BRIGHTNESS**) the pixel is promoted to a strong edge and the **d_changed** flag is set to true to indicate that a change occurred, triggering another iteration.
### Gaussian Filter

### Min Max

### Normalize

### Canny Device
The **cannyDevice** function implements the full Canny edge detection pipeline in the GPU using CUDA. It starts by allocating memory in the divice for the input image **d_idata**, the output data **d_odata**, gradient componentes
## Tests