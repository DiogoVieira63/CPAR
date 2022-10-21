#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#define N 10000000
//#define N 100


#define K 4

float *A, *B, *C,**KL;

int size[K];

void alloc() {
    A = (float *) malloc(N*2*sizeof(float));
    B = (float *) malloc(K*2*sizeof(float));
    C = (float *) malloc(K*2*sizeof(float));


    KL = (float **)malloc(K * sizeof(float *));
    for (int i = 0; i < K;i++){
        KL[i] = (float *) malloc(N*2*sizeof(float));
    }
}


void inicializa() {
    srand(10);

    // initialize vector with N values (x,y)
    for(int i = 0; i < N; i++) {
        A[i*2]   = (float) rand() / RAND_MAX;
        A[i*2+1] = (float) rand() / RAND_MAX;
    }
    // initialize cluster values
    for(int i = 0; i < K; i++) {
        B[i*2]  = A[i*2];
        B[i*2+1]= A[i*2+1];
    }
}


float distance (float *vec, float *cl){
    return sqrt(pow(cl[0] - vec[0],2) + pow(cl[1] - vec[1],2));
}

void addToCluster(int numCluster, float x, float y){
    int n = size[numCluster]++;
    KL[numCluster][n*2]= x;
    KL[numCluster][n*2+1]= y; 
    C[numCluster*2] += x; 
    C[numCluster*2+1] += y; 
}



void k_means(){
    for (int i =0; i < 39;i++){
        printf("-> %d\n",i);
        for (int i = 0; i< N;i++){
            float lowest = distance(&A[i*2],B);
            int index_low = 0;
            for (int k = 1; k < K;k++){
                float dist = distance(&A[i*2],&B[k*2]);
                if (dist < lowest){
                    lowest = dist;
                    index_low = k;
                }
            }
            addToCluster(index_low,A[i*2],A[i*2+1]);
            //printf("%d %f | %f - %f \n",i,lowest,A[2*i],A[2*i+1]);
            //for (int k=0; k < K; k++){}
        }
        for (int u = 0; u < K;u++){
            B[u*2] =   C[u*2]  /size[u];
            B[u*2+1] = C[u*2+1]/size[u];
            C[u*2] = 0;
            C[u*2+1] = 0;
        }
    }
    for (int i = 0; i < K;i++){
        printf("%f | %f \n",B[i*2],B[i*2+1]);

    }
}

int main (){
    alloc();
    inicializa();
    k_means();
}