#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>


__global__ void single_convolution() {

}


// Global declarations
FILE * file;
const char * INPUT_FILE_NAME = "lena_salt.png";
const char * OUTPUT_FILE_NAME = "output.png";

int main() {

    int input_x, input_y, input_channel;
    unsigned char * input_data = stbi_load(INPUT_FILE_NAME, &input_x, &input_y, &input_channel, 1); // 1 - for gray-scale


    if (!input_data) {
        std::cout << "ERROR: can't read file\n";
		return 1;
	}

    std::cout << "Loaded image with a width of " << input_x << "px,a height of " << input_y << "px\n";


    int out_x = input_x, out_y = input_y;
    unsigned char * out_data = (unsigned char *)calloc(out_x * out_y, sizeof(unsigned char)); 

    unsigned char * kernel_data;
    cudaMalloc(&kernel_data, out_x * out_y * sizeof(unsigned char));

    // memcpy(out_data, input_data, out_x * out_y * sizeof(unsigned char));
    // for (int i = 0; i < out_x; ++i) {
    //     for (int j = 0; j < out_y; ++j) {
    //         std::cout << out_data[i * out_y + j] << " ";
    //     }
    //     std::cout << "\n";
    // }
    

    stbi_write_png(OUTPUT_FILE_NAME, out_x, out_y, 1, out_data, 0);
    

    stbi_image_free(input_data);
    stbi_image_free(out_data);


    return 0;
}