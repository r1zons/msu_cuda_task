#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>

__global__ void median_filter(unsigned char * input_data, unsigned char * output_data, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;
    int y = idx / width;
    int x = idx % width;

    const int window_size = 3;
    int window_radius = window_size / 2;

    unsigned char window[window_size * window_size];
    int k = 0;
    for (int i = 0; i < window_size; i++) {
        for (int j = 0; j < window_size; j++) {
            int neighbor_x = x + i - window_radius;
            int neighbor_y = y + j - window_radius;
            if (neighbor_x >= 0 && neighbor_x < width && neighbor_y >= 0 && neighbor_y < height) {
                window[k++] = input_data[neighbor_y * width + neighbor_x];
            } else {
                window[k++] = 0;
            }
        }
    }

    // Sort the window
    for (int i = 0; i < window_size * window_size - 1; i++) {
        for (int j = 0; j < window_size * window_size - i - 1; j++) {
            if (window[j] > window[j + 1]) {
                unsigned char temp = window[j];
                window[j] = window[j + 1];
                window[j + 1] = temp;
            }
        }
    }

    // Calculate median
    unsigned char median = 0;
    if (window_size * window_size % 2 == 1) {
        median = window[window_size * window_size / 2];
    } else {
        median = (window[(window_size * window_size / 2) - 1] + window[window_size * window_size / 2]) / 2;
    }

    output_data[idx] = median;
}

int main() {
    const char * input_file_name = "lena_salt.png";
    const char * output_file_name = "output.png";

    int input_x, input_y, input_channel;
    unsigned char * input_data = stbi_load(input_file_name, &input_x, &input_y, &input_channel, 1); // 1 - for gray-scale

    if (!input_data) {
        std::cout << "ERROR: can't read file\n";
        return 1;
    }

    std::cout << "Loaded image with a width of " << input_x << "px, a height of " << input_y << "px\n";

    int out_x = input_x, out_y = input_y;
    unsigned char * out_data = new unsigned char[out_x * out_y];

    unsigned char * kernel_input_data;
    unsigned char * kernel_output_data;
    cudaMalloc((void **)&kernel_input_data, out_x * out_y * sizeof(unsigned char));
    cudaMalloc((void **)&kernel_output_data, out_x * out_y * sizeof(unsigned char));
    cudaMemcpy(kernel_input_data, input_data, out_x * out_y * sizeof(unsigned char), cudaMemcpyHostToDevice);

    int THREADS = min(256, out_x * out_y);
    int BLOCKS = (out_x * out_y + THREADS - 1) / THREADS;

    dim3 threads(THREADS);
    dim3 blocks(BLOCKS);

    median_filter<<<blocks, threads>>>(kernel_input_data, kernel_output_data, out_x, out_y);

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error!= cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }

    cudaMemcpy(out_data, kernel_output_data, out_x * out_y * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    stbi_write_png(output_file_name, out_x, out_y, 1, out_data, 0);

    cudaFree(kernel_input_data);
    cudaFree(kernel_output_data);
    delete[] out_data;
    stbi_image_free(input_data);

    return 0;
}