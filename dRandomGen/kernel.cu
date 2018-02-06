#include <iostream>

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#include <curand.h>
#include <curand_kernel.h>
using namespace std;

// al
#define N 100
//#define MAX 1

//debug outputs
#define CUDA_KERNEL_DEBUG 0 //test for illegal memory access
#define OUTPUT_PRE 1 // preprocess debug output
#define OUTPUT_POST 1 //postprocess debyg output


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort = true)
{
	if (code != cudaSuccess) {
		std::cout << "GPUassert: " << cudaGetErrorString(code) << " / " << file << " " << line << std::endl;
		//fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
	else {
		if (CUDA_KERNEL_DEBUG == 1) {
			std::cout << "success GPUassert: " << cudaGetErrorString(code) << " / " << file << " " << line << std::endl;
		}
	}
}


// === function definition begins here ===

__global__ void add(int* a, int* b, int* c) { // global function for addition
											  //int N = 4000;
	int bid = blockDim.x * blockIdx.x + threadIdx.x;
	if (bid < N)  // condition of limits
		c[bid] = a[bid] + b[bid];
}

// initialize random numbers
__global__ void init(curandState_t* states, unsigned long seed) {

	int iid = blockDim.x * blockIdx.x + threadIdx.x;
	curand_init(seed, iid, 0, &states[iid]);
}


// generate the random numbers for transition matrix

__global__ void rndn(curandState_t* states, unsigned int* no) { // you can change the data type to unsigned int to give +ve numbers only
	int iid = blockDim.x * blockIdx.x + threadIdx.x;// this is what gives it its dynamic nature, else we use thread
	no[iid] = curand(&states[iid]) % 10;
}

//cudaGetDeviceCount (int* num_devices); // to get the number multiple GPU usable
//cudaSetDevice(int device_id); // to select identifier

// for the status vector
__global__ void s_init(curandState_t* s_states, unsigned long s_seed) {

	int iid = threadIdx.x; // we can also assign as block here? 
	curand_init(s_seed, iid, 0, &s_states[iid]);
}


// generate the random numbers

__global__ void s_rndn (curandState_t* s_states, unsigned int* s_no) { // you can change the data type to unsigned int to give +ve numbers only
	int iid = blockIdx.x;// this is what gives it its dynamic nature, else we use thread
	s_no[iid] = curand(&s_states[iid]) % 2;
}

int main(void) {

	//int N = 4000;
	curandState_t* devState;
	curandState_t* s_devState;
	int j, a[N], b[N], c[N];
	int *dev_a, *dev_b, *dev_c;
	unsigned int s_cpu[N]; // for status vector
	unsigned int* s_gpu; // for status vector
	unsigned int cpu_no[N]; // for random txm
	unsigned int* gpu_no; // fotr random txm

	//allocate memory for random and static state on device
	gpuErrchk(cudaMalloc((void**)&devState, N * sizeof(curandState_t)));
	gpuErrchk(cudaMalloc((void**)&s_devState, N * sizeof(curandState_t)));

	//invoke the initialization/seed for random  txm and static vector

	init << < N, 10 >> > (devState, time(0));
	s_init << < N, 10 >> > (s_devState, time(0));



	// allocate memory for random numbers on device/GPU

	gpuErrchk(cudaMalloc((void**)&gpu_no, N * sizeof(unsigned int)));
	gpuErrchk(cudaMalloc((void**)&s_gpu, N * sizeof(unsigned int)));

	//invoke kernel to launch the random numbers 

	rndn << < N, 10 >> > (devState, gpu_no);
	s_rndn << < N, 10 >> > (s_devState, s_gpu);




	gpuErrchk(cudaMalloc((void**)&dev_a, N * sizeof(int))); // always wrap your code around error checking to keep things safe and catch errors
	gpuErrchk(cudaMalloc((void**)&dev_b, N * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_c, N * sizeof(int)));

	
	//float No[N]; // this is the copy of the arrays to hold the random numbers
	// float *N2;  // this is a pointer to contain the arrays copied from No


	// copy the a, b, c variables to the device

	gpuErrchk(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));//// always wrap your code around error checking to keep things safe and catch errors
	gpuErrchk(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));
	// launch the kernel for the a, b, c variables in the device

	add << < N, 10 >> > (dev_a, dev_b, dev_c);
	// copy back the results to host
	gpuErrchk(cudaMemcpy(c, dev_c, sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(cpu_no, gpu_no, N * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(s_cpu, s_gpu, N * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	
	
	int count = 0;
	for (j = 0; j < N; j++) {

		cout << "the original random numbers generated are :" << cpu_no[j] << 
			" and the status vector random number is :" << s_cpu[j] << "  and the serial number is : " << ++count << endl;
		

		// perform the operation using a[j]
		//cpu_no[j] = j + cpu_no[j];

		//a[j] = cpu_no[j]; // this copies it back to a after operation

		//cout << "this is the value of 'a' after operation with the random numbers: " << a[j] << endl;

		//// perform another operation for b[j]

		//cpu_no[j] = j*j* cpu_no[j];

		//b[j] = cpu_no[j];

		//cout << "this is the value of 'b' after the operation with the random numbers :" << b[j] << endl;

		//c[j] = a[j] + b[j];

		//cout << "the  random numbers after operations is :" << cpu_no[j] << endl;
		//cout << " the result of the additions after the operations is : " << c[j] << endl;
		//cout << " and the count of the number is :" << ++count << endl;
	}


	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	cudaFree(gpu_no);
	cudaFree(devState);
	cudaFree(s_devState);

	cin.get();
	cin.get();
	return 0;
}