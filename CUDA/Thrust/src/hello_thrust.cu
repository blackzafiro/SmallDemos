// Inspired from
// https://developer.nvidia.com/thrust

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <cstdlib>
#include <iostream>

int main(void)
{
	// generate 32M random numbers on the host
	thrust::host_vector<int> h_vec(32 << 20);
	thrust::generate(h_vec.begin(), h_vec.end(), rand);

	// show first 10 elements
	std::cout << "First unsorted (from 32M numbers):" << std::endl;
	for(int i=0; i<10; i++) std::cout << h_vec[i] << "\t"; std::cout << std::endl;

	// transfer data to device
	thrust::device_vector<int> d_vec = h_vec;

	// sort data on the device
	thrust::sort(d_vec.begin(), d_vec.end());

	// transfer data back to host
	thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());

	std::cout << "First sorted:" << std::endl;
	for(int i=0; i<10; i++) std::cout << h_vec[i] << "\t"; std::cout << std::endl;


	return 0;
}
