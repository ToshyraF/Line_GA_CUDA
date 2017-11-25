#include <stdio.h>
#include <stdlib.h>
#include <curand_kernel.h>
#define RANDOM_NUM ((float)rand()/(RAND_MAX))
#define CROSSOVER_RATE 0.7
#define LENGTH 8
#define N 100
struct Population
{
	int m,x,b;
	float FN; 
	float ratecross;
	float ratemutate;
};
int randomPop(){
	int pop=0;
	for(int i =0; i < LENGTH; i++){
		if(RANDOM_NUM > 0.5f){
			pop = pop<<1 | 1;
		}else{
			pop = pop << 1;
		}
	}
	return pop;
}
__device__ float fitness(Population pop,int y0){
	unsigned int y1,err;
	float FN;
	y1 = pop.x*pop.m + pop.b;
	err = y0 - y1;
	// err *= err;
	FN = 1+err;
	FN = 1/FN;
	return FN;

}

__global__ void calculate(Population *pop){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int y0 = pop[0].x*pop[0].m + pop[0].b;
	if(tid < N){
		pop[tid].FN = fitness(pop[tid],y0);
	}
}

__global__ void select(Population *pop,Population *out){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if(tid < N){
		if(pop[tid].FN == 1){
			out[tid] = pop[tid];
			// count = count + 1;
		}
	}
}
__global__ void crossover(Population *pop){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if(tid < N){
			int cross = (int)(pop[tid].ratecross*LENGTH);

			int forward=0;
			int invert=0;
			for(int i=0;i<cross;i++){
				forward = forward<<1 | 1;
			}
			invert = forward ^ 15;
			if(tid == N-1 || pop[tid].FN == 1){
				pop[tid] = pop[tid];
			}
			else{
				pop[tid].m = ((pop[tid].m&forward)|(pop[tid-1].m&invert));
				pop[tid].x = ((pop[tid].x&forward)|(pop[tid-1].x&invert));
				pop[tid].b = ((pop[tid].b&forward)|(pop[tid-1].b&invert));

			}
	}
}
__global__ void mutate(Population *pop){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if(tid < N){
		int cross = (int)(pop[tid].ratemutate*LENGTH);
		int temp=1;
		for(int i=0;i<cross;i++){
			temp = temp<<1;
		}
		if(tid == N-1 || pop[tid].FN == 1){
				pop[tid] = pop[tid];
			}
		else{
			if(((pop[tid].m & temp) == 0)||((pop[tid].x & temp) == 0)||((pop[tid].m & temp) == 0)){
				pop[tid].m = pop[tid].m | temp;
				pop[tid].x = pop[tid].x | temp;
				pop[tid].b = pop[tid].b | temp;
			}
			else
			{
				pop[tid].m = pop[tid].m & (~temp & 15);
				pop[tid].x = pop[tid].x & (~temp & 15);
				pop[tid].b = pop[tid].b & (~temp & 15);
			}
			
		}
	}
}
int main(){
	srand((int)time(NULL));
	Population pop[N],out[N];

	Population *d_pop,*d_out;
	// int x[]={125,12,125,13,12,89};
	// int m[]={123,13,123,56,13,72};
	// int b[]={45,85,45,12,85,64};
	size_t size = N*sizeof(Population);
	//
	for(int i=0; i < N; i++){
         // pop[i].x = x[i];
         // pop[i].m = m[i];
         // pop[i].b = b[i];
		 pop[i].x = randomPop();
         pop[i].m = randomPop();
         pop[i].b = randomPop();
         pop[i].ratecross = RANDOM_NUM;
         pop[i].ratemutate = RANDOM_NUM;
	}

	for(int i=0;i<N;i++){
		printf("input m: %d x: %d b: %d \n",pop[i].m,pop[i].x,pop[i].b);
	}
	cudaMalloc((void **)&d_pop,size);
	cudaMalloc((void **)&d_out,size);

	cudaMemcpy(d_pop,pop,size,cudaMemcpyHostToDevice);
	for(int i=0;i<100000;i++){

		calculate<<<1,N>>>(d_pop);

		crossover<<<1,N>>>(d_pop);

		mutate<<<1,N>>>(d_pop);

		calculate<<<1,N>>>(d_pop);
	}
	cudaMemcpy(out,d_pop,size,cudaMemcpyDeviceToHost);

	for(int i=0;i<N;i++){
		printf("output m: %d x: %d b: %d FN:%f result: %d\n",out[i].m,out[i].x,out[i].b,out[i].FN,(out[i].m*out[i].x)+out[i].b);
		printf("option ratecross: %f ratemutate: %f\n",out[i].ratecross,out[i].ratemutate);
	}
	cudaFree(d_pop);
		cudaFree(d_out);
	return 0;
}