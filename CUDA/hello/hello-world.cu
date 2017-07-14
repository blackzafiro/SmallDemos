// This is the REAL "hello world" for CUDA!
// It takes the string "Hello ", prints it, then passes it to CUDA
// with an array of offsets. Then the offsets are added in parallel
// to produce the string "World!"
// By Ingemar Ragnemalm 2010

// nvcc hello-world.cu -L /usr/local/cuda/lib -lcudart -o hello-world

#include <stdio.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __func__, __LINE__, false); }
inline void gpuAssert(cudaError_t code, const char *file, const char *func, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"\e[1;31mGPUassert: %s %s %s %d\n\e[1;0m", cudaGetErrorString(code), file, func, line);
      if (abort) exit(code);
   }
}

const int N = 16; 
const int blocksize = 16; 

__global__ 
void hello(char *a, int *b) 
{
	// Calculate "world"
	a[threadIdx.x] += b[threadIdx.x];

	// Say where we are working from
        int i = threadIdx.x + blockDim.x * blockIdx.x;
	printf("Hello thread, world from the device number %i \n", i);
}

int main()
{
	char a[N] = "Hello \0\0\0\0\0\0";
	int b[N] = {15, 10, 6, 0, -11, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

	char *ad;
	int *bd;
	const int csize = N*sizeof(char);
	const int isize = N*sizeof(int);

	printf("%s", a);

	gpuErrchk( cudaMalloc( (void**)&ad, csize ) );
	gpuErrchk( cudaMalloc( (void**)&bd, isize ) );
	gpuErrchk( cudaMemcpy( ad, a, csize, cudaMemcpyHostToDevice ) );
	gpuErrchk( cudaMemcpy( bd, b, isize, cudaMemcpyHostToDevice ) );
	
	dim3 dimBlock( blocksize, 1 );
	dim3 dimGrid( 1, 1 );

	// launch a kernel with dimBlock thread groups, and dimGrid blocks per thread.
	hello<<<dimGrid, dimBlock>>>(ad, bd);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
        
	gpuErrchk( cudaMemcpy( a, ad, csize, cudaMemcpyDeviceToHost ) );
	gpuErrchk( cudaFree( ad ) );
	gpuErrchk( cudaFree( bd ) );
	
	printf("%s\n", a);
	return EXIT_SUCCESS;
}
