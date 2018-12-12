#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

#define N 1024

using namespace std;

__global__ void add(int *a, int *b, int *c) {
	c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

void random_ints(int* a, int m)
{
	int i;
	for (i = 0; i < m; ++i)
		a[i] = rand()%5000;
}

int main() {
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

	add << <N, 1 >> > (d_a, d_b, d_c);
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; ++i) {
		cout << a[i] << "+" << b[i] << "=" << c[i] << endl;
	}
	

	delete[] a;
	delete[] b;
	delete[] c;
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	system("PAUSE");
	return 0;
}
