#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

using std::cout;
using std::generate;
using std::vector;

__global__ void jacobi_iteration(const float *A, const float *b, float *x, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            if (j!= idx) {
                sum += A[idx * N + j] * x[j];
            }
        }
        x[idx] = (b[idx] - sum) / A[idx * N + idx];
    }
}

void verify_result(vector<float> &x, vector<float> &b, vector<float> &A, int N) {
    for (int i = 0; i < N; i++) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            sum += A[i * N + j] * x[j];
        }
        assert(fabs(sum - b[i]) < 1e-3);
    }
}

float get_random_num() {
    return static_cast<float>(rand() % 100) / 10.0f;
}

int get_random_sign() {
    return 1 ? (rand() % 100) & 1 : -1;
}

int main() {
    int N = 1 << 5; // Matrix size
    size_t bytes = N * N * sizeof(float);

    vector<float> h_A(N * N);
    vector<float> h_b(N);
    vector<float> h_x(N);

    // Initialize matrices
    generate(h_A.begin(), h_A.end(), []() { return get_random_sign() * get_random_num(); });
    generate(h_b.begin(), h_b.end(), []() { return get_random_sign() * get_random_num(); });

    // Strict-Diagonal dominance
    for (int row = 0; row < N; ++row) {

        float abs_sum = 0.f;
        for (int col = 0; col < N; ++col) {
        if (row == col) continue;

        abs_sum += fabs(h_A[row * N + col]);
        }

        h_A[row * N + row] = abs_sum + 1 + get_random_num();
    }

    // Allocate device memory
    float *d_A, *d_b, *d_x;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_x, N * sizeof(float));

    // Copy data to the device
    cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    int THREADS = 256;
    int BLOCKS = (N + THREADS - 1) / THREADS;

    dim3 threads(THREADS);
    dim3 blocks(BLOCKS);

    // Initialize x to 0
    cudaMemset(d_x, 0, N * sizeof(float));

    // Perform Jacobi iterations
    for (int iter = 0; iter < 10000; iter++) { // adjust the number of iterations as needed
        jacobi_iteration<<<blocks, threads>>>(d_A, d_b, d_x, N);
    }

    // Copy back to the host
    cudaMemcpy(h_x.data(), d_x, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Check result
    verify_result(h_x, h_b, h_A, N);

    cout << "COMPLETED SUCCESSFULLY\n";

    cout << "A:\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
        cout << h_A[i * N + j] << " ";
        }
        cout << "\n";
    }
    cout << "\n";

    cout << "b:\n";
    for (int i = 0; i < N; ++i) {
        cout << h_b[i] << " ";
    }
    cout << "\n";

    cout << "x:\n";
    for (int i = 0; i < N; ++i) {
        cout << h_x[i] << " ";
    }
    cout << "\n";

    // Free memory on device
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_x);

    return 0;
}