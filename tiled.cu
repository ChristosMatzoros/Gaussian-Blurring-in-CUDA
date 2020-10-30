/*
* This sample implements a separable convolution of a 
* 2D image with an arbitrary filter using multiple block_size
* and using the padding technique to avoid divergance. 
* It uses internal padding where its thread computes the convolution
* for one or more pixels of the image. Further more it implements external 
* tiling with multiple kernel invocations. 
* 
*/

//import libraries 
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "gputimer.h"

//Uncomment this to run the CPU implementation and compare the GPU's result with the CPU's result 
//#define ENABLE_PROFILER_STATS

//Timer variables
GpuTimer timer;
double overall_GPU_time = 0;
clock_t start, end;
double overall_CPU_time;

#define IMAGE_DIM (8192*2)	//width size of the image
#define BLOCKED_TILE 1024	//size of the block of tile
#define FILTER_RAD 4		//radius of the filter

#define K 2 				//it defines the internal tiling within each block
#define INTERNAL_TILE_WIDTH	((BLOCKED_TILE)<((32*K)+1) ? (BLOCKED_TILE) : (32*K))
#define N	(IMAGE_DIM/BLOCKED_TILE)
#define BLOCK_WIDTH (INTERNAL_TILE_WIDTH/K)
#define PADDED_WIDTH ((BLOCKED_TILE + 2 * FILTER_RAD)+2*ALIGNMENT)
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
	int internal_block_iters = K+2*((ALIGNMENT+filterR)/blockDim.x);	//It defines how many pixels will transfer each thread to the shared memory
	double result = 0.0;												//Result variable keeps the result during the convolution for each pixel
	
	__shared__ double input[BLOCK_WIDTH][K*BLOCK_WIDTH+2*(FILTER_RAD+ALIGNMENT)];	//Shared mamory array declaration 
	
	//Thread collaboration to bring data to the shared memory array  
	for(i=0;i<internal_block_iters;i++){
		input[threadIdx.y][threadIdx.x+i*BLOCK_WIDTH] = d_Input[(idy*PADDED_WIDTH)+(idx+i*BLOCK_WIDTH)];
	}
	__syncthreads();	//synchronize tο ensure that the shared memory array has all the data we need before we read them  
	
	//row convolution proccess
	for(j=0;j<K;j++){
		result = 0.0;
		for (k = -filterR; k<=filterR; k++) {
			result+=input[threadIdx.y][k+(j*BLOCK_WIDTH)+(threadIdx.x)+FILTER_RAD+ALIGNMENT]*d_Filter[filterR-k];
		}
		d_Buffer[(idy)*imageW+(idx+(j*BLOCK_WIDTH))] = result;
	}
}


//Convolution by columns on GPU
__global__ void convolutionGPUColumn(double *d_Output, double *d_Buffer, int imageW, int imageH, int filterR) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;					//Thread id in the x direction of the grid
	int idy = INTERNAL_TILE_WIDTH * blockIdx.y + threadIdx.y; 			//Thread id in the y direction of the grid(Internal tiling of the block in y direction)
	int k=0,j,i;														//Variables used as iterators
	int internal_block_iters = K +2*((ALIGNMENT+filterR)/blockDim.y);	//It defines how many pixels will transfer each thread to the shared memory
	double result = 0.0;												//Result variable keeps the result during the convolution for each pixel
	
	__shared__ double input[K*BLOCK_WIDTH+2*(FILTER_RAD+ALIGNMENT)][BLOCK_WIDTH];	//Shared mamory array declaration 
	
	//Thread collaboration to bring data to the shared memory array
	for(i=0;i<internal_block_iters;i++){
		input[threadIdx.y+i*BLOCK_WIDTH][threadIdx.x] = d_Buffer[((idy+(i*BLOCK_WIDTH))*BLOCKED_TILE)+idx];
	}
	
	__syncthreads();	//synchronize tο ensure that the shared memory array has all the data we need before we read them	
	
	//column convolution proccess
	for(j=0;j<K;j++){
		result = 0.0;
		for(k=-filterR;k<=filterR;k++) {
			result += input[filterR + ALIGNMENT +threadIdx.y +j*BLOCK_WIDTH +k][threadIdx.x]*d_Filter[filterR-k];
		}
		d_Output[((idy+j*BLOCK_WIDTH)*imageW) +idx] = result;
	}
}

//Main program
int main(int argc, char **argv) {
	int imageW;
	int imageH;
	int i,j,k,l;	
	double *h_Filter,*h_Input,*host_Cpu_Input,* host_OutputGPU;
	double *d_Input0, *d_Output0, *d_Buffer0, *d_Input1, *d_Output1, *d_Buffer1;
	double *host_Grid_Input0, *host_Grid_Input1, *grid_buffer, *final, *hostparBuff0, *hostparBuff1;
	
	#ifdef ENABLE_PROFILER_STATS
	double *host_Buffer,* host_OutputCPU;
	#endif
	
	if((IMAGE_DIM<FILTER_LENGTH)||(IMAGE_DIM%2!=0)){
		printf ( "image size should be a power of two and greater than %d\n", FILTER_LENGTH);
		return(-1);
	}
	
	imageH = IMAGE_DIM;
	imageW = IMAGE_DIM;
	int new_width = (imageW+2*FILTER_RAD)+2*ALIGNMENT;
	
	//Allocate host memory
	h_Filter		= (double *)malloc(FILTER_LENGTH * sizeof(double));
	Check_Allocation_Return_Value(h_Filter)
	
	host_Cpu_Input	= (double *)malloc(imageW * imageW * sizeof(double));
	Check_Allocation_Return_Value(host_Cpu_Input)
	
#ifdef ENABLE_PROFILER_STATS
	host_Buffer		= (double *)malloc(imageW * imageW * sizeof(double));
	Check_Allocation_Return_Value(host_Buffer)
	host_OutputCPU	= (double *)malloc(imageW * imageW * sizeof(double));
	Check_Allocation_Return_Value(host_OutputCPU)
#endif
	
	host_OutputGPU	= (double *)malloc(imageW * imageW * sizeof(double));
	Check_Allocation_Return_Value(host_OutputGPU)
	h_Input 		= (double *)malloc(imageH * new_width * sizeof(double));
	Check_Allocation_Return_Value(h_Input)
	
	
	//grid host stream arrays
	host_Grid_Input0 		= (double *)malloc(BLOCKED_TILE * PADDED_WIDTH * sizeof(double));
	Check_Allocation_Return_Value(host_Grid_Input0)
	memset(host_Grid_Input0,0,BLOCKED_TILE * PADDED_WIDTH * sizeof(double));
	
	host_Grid_Input1 		= (double *)malloc(BLOCKED_TILE * PADDED_WIDTH * sizeof(double));
	Check_Allocation_Return_Value(host_Grid_Input1)
	memset(host_Grid_Input1,0,BLOCKED_TILE * PADDED_WIDTH * sizeof(double));

	grid_buffer 		= (double *)malloc(IMAGE_DIM * new_width * sizeof(double));
	Check_Allocation_Return_Value(grid_buffer)
	memset(grid_buffer,0,IMAGE_DIM * new_width * sizeof(double));	
	
	final = (double *)malloc(IMAGE_DIM * IMAGE_DIM * sizeof(double));
	Check_Allocation_Return_Value(final)
	memset(final,0,IMAGE_DIM * IMAGE_DIM * sizeof(double));
	
	hostparBuff0 = (double *)malloc(BLOCKED_TILE * BLOCKED_TILE * sizeof(double));
	hostparBuff1 = (double *)malloc(BLOCKED_TILE * BLOCKED_TILE * sizeof(double));
	
	
	//Allocate device memory
	cudaMalloc((void**)&d_Input0, BLOCKED_TILE * PADDED_WIDTH * sizeof(double));
	cudaCheckError()
	cudaMalloc((void**)&d_Output0, BLOCKED_TILE * BLOCKED_TILE * sizeof(double));
	cudaCheckError()
	cudaMalloc((void**)&d_Buffer0, BLOCKED_TILE * PADDED_WIDTH * sizeof(double));
	cudaCheckError()
	cudaMalloc((void**)&d_Input1, BLOCKED_TILE * PADDED_WIDTH * sizeof(double));
	cudaCheckError()
	cudaMalloc((void**)&d_Output1, BLOCKED_TILE * BLOCKED_TILE * sizeof(double));
	cudaCheckError()
	cudaMalloc((void**)&d_Buffer1, BLOCKED_TILE * PADDED_WIDTH * sizeof(double));
	cudaCheckError()
	
	cudaMemset(d_Buffer0,0,BLOCKED_TILE*PADDED_WIDTH*sizeof(double));
	cudaMemset(d_Output0,0,BLOCKED_TILE*BLOCKED_TILE*sizeof(double));
	cudaMemset(d_Buffer1,0,BLOCKED_TILE*PADDED_WIDTH*sizeof(double));
	cudaMemset(d_Output1,0,BLOCKED_TILE*BLOCKED_TILE*sizeof(double));
	
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
			if((j<FILTER_RAD+ALIGNMENT) || (j>=(imageH+FILTER_RAD+ALIGNMENT))){
				h_Input[i*new_width+j]= 0.0;
			}
			else{
				h_Input[i*new_width+j] = host_Cpu_Input[i*imageH+(j-FILTER_RAD-ALIGNMENT)];
			}
		}
	}
	
	
#ifdef ENABLE_PROFILER_STATS
	///CPU convolution
	start = clock();	//start clock
	convolutionRowCPU(host_Buffer, host_Cpu_Input, h_Filter, imageW, imageH, FILTER_RAD); 
	convolutionColumnCPU(host_OutputCPU, host_Buffer, h_Filter, imageW, imageH, FILTER_RAD); 
	end = clock();		//end clock
	
#endif
	
	timer.Start();
	cudaMemcpyToSymbol( d_Filter, h_Filter, FILTER_LENGTH * sizeof(double),0,cudaMemcpyHostToDevice);
	cudaCheckError()
	timer.Stop();
	overall_GPU_time +=timer.Elapsed();
	
	dim3 dimBlock;
	dim3 dimGrid;
	dimBlock.x =BLOCK_WIDTH;
	dimBlock.y =BLOCK_WIDTH;
	
	dimGrid.x = BLOCKED_TILE/INTERNAL_TILE_WIDTH;
	dimGrid.y = BLOCKED_TILE/BLOCK_WIDTH;
	
	timer.Start();
	for(k=0;k<N;k++){
		for(l=0;l<N;l+=2){
			
			for(i=0;i<BLOCKED_TILE;i++){
				for(j=0;j<PADDED_WIDTH;j++){
					host_Grid_Input0[i*PADDED_WIDTH+j] = h_Input[((k*BLOCKED_TILE)+i)*(IMAGE_DIM+2*FILTER_RAD+2*ALIGNMENT) + (BLOCKED_TILE*l)+j];
					host_Grid_Input1[i*PADDED_WIDTH+j] = h_Input[((k*BLOCKED_TILE)+i)*(IMAGE_DIM+2*FILTER_RAD+2*ALIGNMENT) + (BLOCKED_TILE*(l+1))+j];
				}
			}
			
			//Copy host array to device0
			cudaMemcpy(d_Input0, host_Grid_Input0, BLOCKED_TILE * PADDED_WIDTH * sizeof(double),cudaMemcpyHostToDevice);
			cudaCheckError()
			//Copy host array to device1
			cudaMemcpy(d_Input1, host_Grid_Input1, BLOCKED_TILE * PADDED_WIDTH * sizeof(double),cudaMemcpyHostToDevice);
			cudaCheckError()
			
			
			//convolutionGPURow First invocation
			convolutionGPURow<<<dimGrid,dimBlock>>>(d_Buffer0, d_Input0, BLOCKED_TILE, BLOCKED_TILE, FILTER_RAD);
			cudaDeviceSynchronize();
			cudaCheckError()
			
			//convolutionGPURow Second invocation
			convolutionGPURow<<<dimGrid,dimBlock>>>(d_Buffer1, d_Input1, BLOCKED_TILE, BLOCKED_TILE, FILTER_RAD);
			cudaDeviceSynchronize();
			cudaCheckError()
		
			
			//Buffer0 copy device to host
			cudaMemcpy(hostparBuff0, d_Buffer0, BLOCKED_TILE* BLOCKED_TILE * sizeof(double), cudaMemcpyDeviceToHost);
			cudaCheckError()
			
			//Buffer1 copy device to host
			cudaMemcpy(hostparBuff1, d_Buffer1, BLOCKED_TILE* BLOCKED_TILE * sizeof(double), cudaMemcpyDeviceToHost);			
			cudaCheckError()
			
			for(i=0;i<BLOCKED_TILE;i++){
				for(j=0;j<BLOCKED_TILE;j++){
					grid_buffer[((i*IMAGE_DIM+k*BLOCKED_TILE*IMAGE_DIM)+j+l*BLOCKED_TILE)+IMAGE_DIM*(FILTER_RAD+ALIGNMENT)] = hostparBuff0[i*BLOCKED_TILE+j];
					grid_buffer[((i*IMAGE_DIM+k*BLOCKED_TILE*IMAGE_DIM)+j+(l+1)*BLOCKED_TILE)+IMAGE_DIM*(FILTER_RAD+ALIGNMENT)] = hostparBuff1[i*BLOCKED_TILE+j];
				}
			}
		}
	}
	
	timer.Stop();
	overall_GPU_time += timer.Elapsed();
	
	dimBlock.x =BLOCK_WIDTH;
	dimBlock.y =BLOCK_WIDTH;
	dimGrid.x = BLOCKED_TILE/BLOCK_WIDTH;
	dimGrid.y = BLOCKED_TILE/INTERNAL_TILE_WIDTH;
	
	timer.Start();
	for(k=0;k<N;k++){
		for(l=0;l<N;l+=2){
			
			for(i=0;i<PADDED_WIDTH;i++){
				for(j=0;j<BLOCKED_TILE;j++){					
					host_Grid_Input0[i*BLOCKED_TILE+j] = grid_buffer[((k*BLOCKED_TILE)+i)*(IMAGE_DIM) + (BLOCKED_TILE*l)+j];
				}
			}
	
			//Copy host array to device
			cudaMemcpy(d_Buffer0, host_Grid_Input0, BLOCKED_TILE * PADDED_WIDTH * sizeof(double),cudaMemcpyHostToDevice);
			cudaCheckError()
			
			convolutionGPUColumn<<<dimGrid,dimBlock>>>(d_Output0, d_Buffer0,BLOCKED_TILE, BLOCKED_TILE, FILTER_RAD);
			cudaDeviceSynchronize();
			cudaCheckError()
			
			//Buffers copy device to host
			cudaMemcpy(hostparBuff0, d_Output0, BLOCKED_TILE* BLOCKED_TILE * sizeof(double), cudaMemcpyDeviceToHost);
			cudaCheckError()
			
			for(i=0;i<BLOCKED_TILE;i++){
				for(j=0;j<BLOCKED_TILE;j++){
					final[((i*IMAGE_DIM+k*BLOCKED_TILE*IMAGE_DIM)+j+l*BLOCKED_TILE)] = hostparBuff0[i*BLOCKED_TILE+j];
				}
			}
		
			for(i=0;i<PADDED_WIDTH;i++){
				for(j=0;j<BLOCKED_TILE;j++){
					host_Grid_Input1[i*BLOCKED_TILE+j] = grid_buffer[((k*BLOCKED_TILE)+i)*(IMAGE_DIM) + (BLOCKED_TILE*(l+1))+j];
				}
			}
			
			//Copy host array to device
			cudaMemcpy(d_Buffer1, host_Grid_Input1, BLOCKED_TILE * PADDED_WIDTH * sizeof(double),cudaMemcpyHostToDevice);
			cudaCheckError()
			
			convolutionGPUColumn<<<dimGrid,dimBlock>>>(d_Output1, d_Buffer1,BLOCKED_TILE, BLOCKED_TILE, FILTER_RAD);
			cudaDeviceSynchronize();
			cudaCheckError()
		
			cudaMemcpy(hostparBuff1, d_Output1, BLOCKED_TILE* BLOCKED_TILE * sizeof(double), cudaMemcpyDeviceToHost);			
			cudaCheckError()
			
			for(i=0;i<BLOCKED_TILE;i++){
				for(j=0;j<BLOCKED_TILE;j++){
					final[((i*IMAGE_DIM+k*BLOCKED_TILE*IMAGE_DIM)+j+(l+1)*BLOCKED_TILE)] = hostparBuff1[i*BLOCKED_TILE+j];
				}
			}
		}
	}
	timer.Stop();
	overall_GPU_time += timer.Elapsed();
	
	
#ifdef ENABLE_PROFILER_STATS
	//Check for errors between the CPU implementation and the GPU implementation
	
	double difference=0,max_difference=0;
	
	for (i = 0; i < imageW * imageH; i++){
		difference = ABS(host_OutputCPU[i] - final[i]);
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
	overall_CPU_time = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC ;
	printf("Time elapsed on CPU =                   %gms\n", overall_CPU_time);
#endif
	printf("Time elapsed on GPU(overall) =           %gms\n", overall_GPU_time);
	
	//free Device memory
	cudaFree(d_Input0);
	cudaCheckError()
	cudaFree(d_Output0);
	cudaCheckError()
	cudaFree(d_Buffer0);
	cudaCheckError()
	cudaFree(d_Input1);
	cudaCheckError()
	cudaFree(d_Output1);
	cudaCheckError()
	cudaFree(d_Buffer1);
	cudaCheckError()
	
	//free host memory
	free(host_Grid_Input0);
	free(host_Grid_Input1);
	free(h_Input);
	free(h_Filter);
	free(host_Cpu_Input);
	free(grid_buffer);
	free(final);
	free(host_OutputGPU);
	free(hostparBuff0);
	free(hostparBuff1);
#ifdef ENABLE_PROFILER_STATS
	free(host_Buffer);
	free(host_OutputCPU);
#endif
	
	
	//reset device
	cudaDeviceReset();
	
	return(0);
}