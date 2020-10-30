/*
* This sample implements a separable convolution of a 
* 2D image with an arbitrary filter using multiple block_size
* and using the padding technique to avoid divergance. 
* It uses internal padding where its thread computes the convolution
* for one or more pixels of the image.
*/


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "gputimer.h"

GpuTimer timer;
double overall_GPU_time = 0, overall_data_transfer_time = 0;
clock_t start, end;
double overall_CPU_time;

int filter_radius;

//Change these two parameters 
#define IMAGE_DIM  8192		//size of the block of tile
#define FILTER_RAD 32		//radius of the filter


#define K 2 				//it defines the internal tiling within each block
#define INTERNAL_TILE_WIDTH	((IMAGE_DIM)<((32*K)+1) ? (IMAGE_DIM) : (32*K))
#define BLOCK_WIDTH (INTERNAL_TILE_WIDTH/K)
#define PADDED_WIDTH ((IMAGE_DIM + 2 * FILTER_RAD)+2*ALIGNMENT)
#define FILTER_LENGTH 	(2 * FILTER_RAD + 1)
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define accuracy  	0.00005
#define PAD (FILTER_RAD)
#define ALIGNMENT  (BLOCK_WIDTH-(FILTER_RAD%BLOCK_WIDTH))


//Error handling using functions of the CUDA runtime API
#define cudaCheckError() {																\
	cudaError_t e=cudaGetLastError();													\
	if(e!=cudaSuccess) {																\
		printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));	\
		exit(EXIT_FAILURE);																\
	}																					\
}

//This Macro checks malloc() and cudaMalloc() return values
#define Check_Allocation_Return_Value(a){		\
	if(a==NULL) {								\
		printf("Allocation Error\n");			\
		exit(EXIT_FAILURE);						\
	}											\
}

// Reference row convolution filter
void convolutionRowCPU(double *h_Dst, double *h_Src, double *h_Filter, int imageW, int imageH, int filterR) {
	int x, y, k;
	
	
	for (y = 0; y < imageH; y++) {
		for (x = 0; x < imageW; x++) {
			double sum = 0;
			for (k = -filterR; k <= filterR; k++) {
				int d = x + k;
				if (d >= 0 && d < imageW) {
					sum += h_Src[y * imageW + d] * h_Filter[filterR - k];
				}
			h_Dst[y * imageW + x] = sum;
			}
		}
	}
}



// Reference column convolution filter
void convolutionColumnCPU(double *h_Dst, double *h_Src, double *h_Filter, int imageW, int imageH, int filterR) {
	int x, y, k;
	
	for (y = 0; y < imageH; y++) {
		for (x = 0; x < imageW; x++) {
			double sum = 0;
			for (k = -filterR; k <= filterR; k++) {
				int d = y + k;
				if (d >= 0 && d < imageH) {
					sum += h_Src[d * imageW + x] * h_Filter[filterR - k];
				}
				h_Dst[y * imageW + x] = sum;
			}
		}
	}
}
__constant__ double d_Filter[FILTER_LENGTH];

//Convolution by rows on GPU
__global__ void convolutionGPURow(double *d_Buffer, double *d_Input, int imageW, int imageH, int filterR) {
	int idx = blockIdx.x * INTERNAL_TILE_WIDTH + threadIdx.x;			//Thread id in the x direction of the grid(Internal tiling of the block in x direction)
	int idy = blockIdx.y * blockDim.y + threadIdx.y;					//Thread id in the y direction of the grid
	int k=0,i=0,j;														//Variables used as iterators
	int iters = K +2*((ALIGNMENT+filterR)/blockDim.x);					//It defines how many pixels will transfer each thread to the shared memory
	double sum;															//Result variable keeps the result during the convolution for each pixel
	
	__shared__ double input[BLOCK_WIDTH][K*BLOCK_WIDTH+2*(FILTER_RAD+ALIGNMENT)];	//Shared mamory array declaration
	
	//Thread collaboration to bring data to the shared memory array  
	for(i=0;i<iters;i++){
		input[threadIdx.y][threadIdx.x+i*BLOCK_WIDTH] = d_Input[(idy*PADDED_WIDTH)+(idx+i*BLOCK_WIDTH)];
	}
	
	__syncthreads();	//synchronize tο ensure that the shared memory array has all the data we need before we read them  

	sum = 0.0;
	//row convolution proccess
	for(j=0;j<K;j++){
		sum = 0.0;
		for (k = -filterR; k<=filterR; k++) {
			sum+=input[threadIdx.y][k+(j*BLOCK_WIDTH)+(threadIdx.x)+FILTER_RAD+ALIGNMENT]*d_Filter[filterR-k];
		}
		d_Buffer[(idy+filterR+ALIGNMENT)*imageW+(idx+(j*BLOCK_WIDTH))] = sum;
	}
}

//Convolution by columns on GPU
__global__ void convolutionGPUColumn(double *d_Output, double *d_Buffer, int imageW, int imageH, int filterR) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x; 					//Thread id in the x direction of the grid
	int idy = INTERNAL_TILE_WIDTH * blockIdx.y + threadIdx.y; 			//Thread id in the y direction of the grid(Internal tiling of the block in y direction)
	int k=0,j,i;														//Variables used as iterators
	int iters = K +2*((ALIGNMENT+filterR)/blockDim.y);					//It defines how many pixels will transfer each thread to the shared memory
	double sum;
	
	
	__shared__ double input[K*BLOCK_WIDTH+2*(FILTER_RAD+ALIGNMENT)][BLOCK_WIDTH];	//Shared mamory array declaration
	
	//Thread collaboration to bring data to the shared memory array
	for(i=0;i<iters;i++){
		input[threadIdx.y+i*BLOCK_WIDTH][threadIdx.x] = d_Buffer[((idy+(i*BLOCK_WIDTH))*IMAGE_DIM)+idx];
	}
	
	__syncthreads();	//synchronize tο ensure that the shared memory array has all the data we need before we read them	
	
	sum = 0.0;
	//column convolution proccess
	for(j=0;j<K;j++){
		sum = 0.0;
		for(k=-filterR;k<=filterR;k++) {
			sum += input[filterR + ALIGNMENT +threadIdx.y +j*BLOCK_WIDTH +k][threadIdx.x]*d_Filter[filterR-k];
		}
		d_Output[((idy+j*BLOCK_WIDTH)*imageW) +idx] = sum;
	}
}

//Main program
int main(int argc, char **argv) {
	int imageW;
	int imageH;
	int i,j;	
	double *h_Filter,*h_Input,*host_Cpu_Input,*host_Buffer,* host_OutputCPU,* host_OutputGPU;
	double * d_Input,* d_Output,* d_Buffer;
	
	int side_length	= IMAGE_DIM;
	filter_radius 	= FILTER_RAD;
	
	if((side_length<FILTER_LENGTH)||(side_length%2!=0)){
		printf ( "image size should be a power of two and greater than %d\n", FILTER_LENGTH);
		return(-1);
	}
	
	imageH = side_length;
	imageW = side_length;
	int new_width = (imageW+2*filter_radius)+2*ALIGNMENT;
	
	h_Input 		= (double *)malloc(imageH * new_width * sizeof(double));
	Check_Allocation_Return_Value(h_Input)
	h_Filter		= (double *)malloc(FILTER_LENGTH * sizeof(double));
	Check_Allocation_Return_Value(h_Filter)
	host_OutputGPU	= (double *)malloc(imageW * imageW * sizeof(double));
	Check_Allocation_Return_Value(host_OutputGPU)
	host_Cpu_Input	= (double *)malloc(imageW * imageW * sizeof(double));
	Check_Allocation_Return_Value(host_Cpu_Input)
	host_Buffer		= (double *)malloc(imageW * imageW * sizeof(double));
	Check_Allocation_Return_Value(host_Buffer)
	host_OutputCPU	= (double *)malloc(imageW * imageW * sizeof(double));
	Check_Allocation_Return_Value(host_OutputCPU)
	
	
	//Allocate device memory
	d_Input = NULL;
	cudaMalloc((void**)&d_Input, imageH * new_width * sizeof(double));
	cudaCheckError()
	d_Output = NULL;
	cudaMalloc((void**)&d_Output, imageH * imageW * sizeof(double));
	cudaCheckError()
	d_Buffer = NULL;
	cudaMalloc((void**)&d_Buffer, imageW * new_width * sizeof(double));
	cudaCheckError()
	
	cudaMemset(d_Buffer,0,new_width*imageW*sizeof(double));
	cudaMemset(d_Output,0,imageW*imageH*sizeof(double));
	
	srand(200);
	for (i = 0; i < FILTER_LENGTH; i++) {
		h_Filter[i] = (double)(rand() % 16);
	}
	for (i = 0; i<imageH; i++) {
		for(j=0; j<imageW; j++){
			host_Cpu_Input[i*imageH+j] = (double)rand() / ((double)RAND_MAX / 255) + (double)rand() / (double)RAND_MAX;
		}
	}

	//image "sinking"
	for(i = 0; i<imageW; i++) {
		for(j=0; j<new_width;  j++){
			if((j<filter_radius+ALIGNMENT) || (j>=(imageH+filter_radius+ALIGNMENT))){
				h_Input[i*new_width+j]= 0.0;
			}
			else{
				h_Input[i*new_width+j] = host_Cpu_Input[i*imageH+(j-filter_radius-ALIGNMENT)];
			}
		}
	}
	
	///CPU convolution
	start = clock();	//start clock
	convolutionRowCPU(host_Buffer, host_Cpu_Input, h_Filter, imageW, imageH, filter_radius); // convolution kata grammes

	convolutionColumnCPU(host_OutputCPU, host_Buffer, h_Filter, imageW, imageH, filter_radius); // convolution kata sthles
	end = clock();		//end clock
	
	overall_CPU_time = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC ;
	printf("Time elapsed on CPU =                   %gms\n", overall_CPU_time);
	
	//copy host memory to device memory
	timer.Start();
	cudaMemcpy(d_Input, h_Input, imageH * new_width * sizeof(double),cudaMemcpyHostToDevice);
	cudaCheckError()
	timer.Stop();
	overall_data_transfer_time +=timer.Elapsed();
	
	timer.Start();
	cudaMemcpyToSymbol( d_Filter, h_Filter, FILTER_LENGTH * sizeof(double),0,cudaMemcpyHostToDevice);
	cudaCheckError()
	timer.Stop();
	overall_data_transfer_time +=timer.Elapsed();
	
	dim3 dimBlock;
	dim3 dimGrid;
	if(imageW*imageH<=1024){
		dimBlock.x =BLOCK_WIDTH;
		dimBlock.y =BLOCK_WIDTH;
		
		dimGrid.x = imageW/INTERNAL_TILE_WIDTH;
		dimGrid.y = imageH/BLOCK_WIDTH;
		
		timer.Start();
		convolutionGPURow<<<dimGrid,dimBlock>>>(d_Buffer, d_Input, imageW, imageH, filter_radius);
		timer.Stop();
		overall_GPU_time += timer.Elapsed();
		cudaDeviceSynchronize();
		cudaCheckError()
		
		dimBlock.x =BLOCK_WIDTH;
		dimBlock.y =BLOCK_WIDTH;
		dimGrid.x = imageH/BLOCK_WIDTH;
		dimGrid.y = imageW/INTERNAL_TILE_WIDTH;
		
		timer.Start();
		convolutionGPUColumn<<<dimGrid,dimBlock>>>(d_Output, d_Buffer,imageW, imageH, filter_radius);
		timer.Stop();
		overall_GPU_time += timer.Elapsed();
		cudaDeviceSynchronize();
		cudaCheckError()
		
	}	
	else{
		dimBlock.x =BLOCK_WIDTH;
		dimBlock.y =BLOCK_WIDTH;
		dimGrid.x = imageW / INTERNAL_TILE_WIDTH;
		dimGrid.y = imageH / BLOCK_WIDTH;
		
		timer.Start();
		convolutionGPURow<<<dimGrid,dimBlock>>>(d_Buffer, d_Input, imageW, imageH, filter_radius);
		timer.Stop();
		overall_GPU_time += timer.Elapsed();
		cudaDeviceSynchronize();
		cudaCheckError()
		
		dimGrid.y = imageH / INTERNAL_TILE_WIDTH;
		dimGrid.x = imageW / BLOCK_WIDTH;
		
		timer.Start();
		convolutionGPUColumn<<<dimGrid,dimBlock>>>(d_Output, d_Buffer, imageW, imageH, filter_radius);
		timer.Stop();
		overall_GPU_time += timer.Elapsed();
		cudaDeviceSynchronize();
		cudaCheckError()
	}
	
	timer.Start();
	cudaMemcpy(host_OutputGPU, d_Output, imageW * imageH * sizeof(double),cudaMemcpyDeviceToHost);
	cudaCheckError()
	timer.Stop();
	overall_data_transfer_time += timer.Elapsed();
	
	double difference=0,max_difference=0;
	
	for (i = 0; i < imageW * imageH; i++){
		difference = ABS(host_OutputCPU[i] - host_OutputGPU[i]);
		if (difference> max_difference) {
			max_difference = difference;
		}
		if (accuracy < max_difference){
			printf("The accuracy(%lf) is not good enough\n",accuracy);
			break;
		}
	}
	
	//for debuging
	printf("filter radius:%d, max_difference: %lf\n",FILTER_RAD,max_difference);	
	
	printf("Time elapsed on GPU(memory transfers) = %gms\n", overall_data_transfer_time);
	printf("Time elapsed on GPU(computation) =      %gms\n", overall_GPU_time);
	printf("Time elapsed on GPU(overall) =           %gms\n", overall_GPU_time + overall_data_transfer_time);
	
	printf("(computation)/(memory transfers):%lf\n",overall_GPU_time/overall_data_transfer_time);
	
	//free Device memory
	cudaFree(d_Input);
	cudaCheckError()
	cudaFree(d_Output);
	cudaCheckError()
	cudaFree(d_Buffer);
	cudaCheckError()
	
	//free host memory
	free(h_Input);
	free(h_Filter);
	free(host_Cpu_Input);
	free(host_Buffer);
	free(host_OutputCPU);
	free(host_OutputGPU);
	
	//reset device
	cudaDeviceReset();
	
	return 0;
}