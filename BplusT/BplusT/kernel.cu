#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cmath>
#include <chrono>
#include <fstream>

#define N (4076*4076)
#define THREADS_PER_BLOCK 512

using namespace std;
using namespace chrono;

__global__ void add(int *a, int *b, int *c) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	c[index] = a[index] + b[index];
}

void random_ints(int* a, int m)
{
	int i;
	for (i = 0; i < m; ++i)
		a[i] = rand() % 5000;
}

int main() {
	ofstream f("numere.txt");
	time_point<steady_clock> start, end;
	int *a, *b, *c;
	int *d_a, *d_b, *d_c;
	int size = sizeof(int)*N;
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);

	a = new int[N];
	random_ints(a, N);
	b = new int[N];
	random_ints(b, N);
	c = new int[N];

	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	start = steady_clock::now();
	add << <N / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (d_a, d_b, d_c);
	end = steady_clock::now();
	duration<double> elapsed_seconds = end - start;
	cout << elapsed_seconds.count() << endl;
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; ++i) {
		f << a[i] << "+" << b[i] << "=" << c[i] << endl;
	}
	//cout << c[1] << endl;

	f.close();

	delete[] a;
	delete[] b;
	delete[] c;
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	system("PAUSE");
	return 0;
}
