#include "stencil.h"
#include <printf.h>
#include <time.h>

#define NUM_BLOCKS 512
#define NUM_THREADS_PER_BLOCK 256
#define SIZE NUM_BLOCKS *NUM_THREADS_PER_BLOCK

#define N 10000000

using namespace std;

#define X(i) (i * 2)
#define Y(i) (i * 2 + 1)

#define K 4

float *pointsX, *pointsY, *centroids;
int *cluster;


/*
__global__
void stencilKernel (float *a, float *c) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	// initialise the array with the results
	c[id] = 0;

	// iterate through the neighbours required to calculate
	// the values for the current position of c
	for (int n = id-2; n <= id+2; n++) {
		if (n > 0 && n < 	) c[id]+= a[n];
	}

}
*/

// Function to calculate the euclidean distance
__device__ float distance(float x, float y, float *cl)
{
	return ((cl[0] - x) * (cl[0] - x) + (cl[1] - y) * (cl[1] - y));
}

/*
__global__ void assignClusterKernel(int N, float *a, float *b, int *c)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
	{
		float lowest = distance(vectorX[(i)], vectorY[(i)], &centroid[X(0)]);
		int index_low = 0;
		for (int k = 1; k < K; k++)
		{
			float dist = distance(vectorX[(i)], vectorY[(i)], &centroid[X(k)]);
			if (dist < lowest)
			{
				lowest = dist;
				index_low = k;
			}
		}
		cluster[i] = index_low;
	}
}

__global__ void sumCentroidsKernel(int N, int *size, float *sumX, float *sumY, float *vectorX, float *vectorY)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
	{
		int index_low = cluster[i];
		sumX += vectorX[(i)];
		sumY += vectorY[(i)];
		size[index_low] += 1;
	}
}

__global__ void calculateCentroidsKernel(int N, int *size, float *sumX, float *sumY, float *vectorX, float *vectorY)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
	{
		int index_low = cluster[i];
		sumX += vectorX[(i)];
		sumY += vectorY[(i)];
		size[index_low] += 1;
	}
}

*/

//__global__ void assignCluster(float *pointX, float *pointY, float *centroids, int *size, float* sum )

__global__ void kmeans(float *pointX, float *pointY, float *centroids,unsigned int *size, float *sum)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Print some debugging information
	if (tid >= N)
		return;
	
	//centroids to shared memory
	__shared__ float sharedCentroids[K * 2];
	if (threadIdx.x < K * 2)
	{
		sharedCentroids[threadIdx.x] = centroids[threadIdx.x];
	}
	__syncthreads();


	// Calculate the distance between the point and each centroid
	float min_distance = distance(pointX[tid], pointY[tid], &sharedCentroids[0]);
	int min_index = 0;
	for (int i = 1; i < K; i++)
	{
		float dist = distance(pointX[tid], pointY[tid], &sharedCentroids[2 * i]);
		if (dist < min_distance)
		{
			min_distance = dist;
			min_index = i;
		}
	}

	// Update the sums and sizes of the cluster using shared memory
	atomicAdd(&sum[X(min_index)], pointX[tid]);
	atomicAdd(&sum[Y(min_index)], pointY[tid]);
	atomicAdd(&size[min_index], 1);
}


__global__ void kmeans2(float *pointX, float *pointY, float *centroids,int *cluster)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Print some debugging information
	if (tid >= N)
		return;
	
	//centroids to shared memory
	__shared__ float sharedCentroids[K * 2];
	if (threadIdx.x < K * 2)
	{
		sharedCentroids[threadIdx.x] = centroids[threadIdx.x];
	}
	__syncthreads();


	// Calculate the distance between the point and each centroid
	float min_distance = distance(pointX[tid], pointY[tid], &sharedCentroids[0]);
	int min_index = 0;
	for (int i = 1; i < K; i++)
	{
		float dist = distance(pointX[tid], pointY[tid], &sharedCentroids[2 * i]);
		if (dist < min_distance)
		{
			min_distance = dist;
			min_index = i;
		}
	}
	cluster[tid] = min_index;

}

void launchStencilKernel()
{
	// pointers to the device memory
	float *dX, *dY, *dC;
	//unsigned int *dSize;
	int *dCluster;

	// declare variable with size of the array in bytes
	int bytes = N * sizeof(float);
	//int bytesInt = K * sizeof(unsigned int);
	int bytesCentroids = K * 2 * sizeof(float);
	int bytesCluster = N * sizeof(int);

	// allocate the memory on the device
	printf("Allocating On Device\n");
	cudaMalloc((void **)&dX, bytes);
	cudaMalloc((void **)&dY, bytes);
	cudaMalloc((void **)&dC, bytesCentroids);
	//cudaMalloc((void **)&dSize, bytesInt);
	//cudaMalloc((void **)&dSum, bytesCentroids);
	cudaMalloc((void **)&dCluster, bytesCluster);

	checkCUDAError("mem allocation");

	// copy inputs to the device
	cudaMemcpy(dX, pointsX, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dY, pointsY, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dC, centroids, bytesCentroids, cudaMemcpyHostToDevice);

	checkCUDAError("memcpy h->d");

	// launch the kernel
	startKernelTime();

	dim3 threads_block(NUM_THREADS_PER_BLOCK);
	int blocks_int = (N + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
	dim3 blocks(blocks_int);
	printf("Launching with %d blocks of %d threads\n", blocks_int, NUM_THREADS_PER_BLOCK);
	int size[K] = {0};
	for (int i = 0; i <= 20; i++)
	{
		// unsigned int size[K] = {0};
		// float sum[K * 2] = {0};
		// cudaMemcpy(dSize, size, bytesInt, cudaMemcpyHostToDevice);
		// cudaMemcpy(dSum, sum, bytesCentroids, cudaMemcpyHostToDevice);
		//kmeans<<<blocks, threads_block>>>(dX, dY, dC, dSize, dSum);
		kmeans2<<<blocks, threads_block>>>(dX, dY, dC, dCluster);

		cudaMemcpy(cluster,dCluster, bytesCluster, cudaMemcpyDeviceToHost);
		memset(size, 0,K*sizeof(int));
		float sum[K * 2] = {0};

		for (int i = 0; i < N; i++){
			sum[X(cluster[i])] += pointsX[i];
			sum[Y(cluster[i])] += pointsY[i];
			size[cluster[i]]++;
		}
		//cudaMemcpy(centroids, dC, bytesCentroids, cudaMemcpyDeviceToHost);
		//cudaMemcpy(size, dSize, bytesInt, cudaMemcpyDeviceToHost);
		//cudaMemcpy(sum, dSum, bytesCentroids, cudaMemcpyDeviceToHost);

		for (int i = 0; i < K; i++){
			centroids[X(i)] = sum[X(i)] / size[i];
			centroids[Y(i)] = sum[Y(i)] / size[i];
		}
		cudaMemcpy(dC, centroids, bytesCentroids, cudaMemcpyHostToDevice);
	}

	stopKernelTime();
	checkCUDAError("kernel invocation");
	//int size[K];

	// copy the output to the host
	//cudaMemcpy(centroids, dC, bytesCentroids, cudaMemcpyDeviceToHost);
	//cudaMemcpy(size, dSize, bytesInt, cudaMemcpyDeviceToHost);
	checkCUDAError("memcpy d->h");

	// print info about centroids and size
	for (int i = 0; i < K; i++)
	{
		printf("Centroid %d: (%f, %f) with size %d\n", i, centroids[X(i)], centroids[Y(i)], size[i]);
	}

	// free the device memory
	cudaFree(dX);
	cudaFree(dY);
	cudaFree(dC);
	cudaFree(dCluster);
	//cudaFree(dSize);
	//cudaFree(dSum);
	checkCUDAError("mem free");
}


int main(int argc, char **argv)
{
	clock_t start, end;
    start = clock();
    double time_used;
	printf("Starting K-means\n");
	// arrays on the host
	
	pointsX = (float *)malloc(N * sizeof(float));
	pointsY = (float *)malloc(N * sizeof(float));
	centroids = (float *)malloc(K * 2 * sizeof(float));
	cluster = (int *)malloc(N * sizeof(int));

	srand(10);

	printf("Initialising arrays with random values between 0 and 1\n");
	// initialises the array
	for (int i = 0; i < N; ++i)
	{
		pointsX[i] = (float)rand() / RAND_MAX;
		pointsY[i] = (float)rand() / RAND_MAX;
	}

	// initialises the centroids
	printf("Initialising centroids with vectors first K values\n");
	for (int i = 0; i < K; ++i)
	{
		centroids[X(i)] = pointsX[i];
		centroids[Y(i)] = pointsY[i];
	}
	launchStencilKernel();
	end = clock();
    time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Execution time: %lf seconds\n", time_used);
	return 0;
}
