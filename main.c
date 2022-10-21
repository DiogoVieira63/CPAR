#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#define N 10000000
//#define N 100


#define K 4

float *A, *B, *C,**KL;

int tempSize[K],size[K];

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
        size[i] = 0;
    }
}


float distance (float *vec, float *cl){
    return sqrt(pow(cl[0] - vec[0],2) + pow(cl[1] - vec[1],2));
}

void addToCluster(int numCluster, float x, float y){
    int n = tempSize[numCluster]++;
    KL[numCluster][n*2]= x;
    KL[numCluster][n*2+1]= y; 
    C[numCluster*2] += x; 
    C[numCluster*2+1] += y; 
}



void k_means(){
    int changed = 1, iterations = 0;
    while (changed){
        for (int i = 0; i < N ;i++){
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
        changed = 0;
        for (int u = 0; u < K;u++){
            B[u*2] =   C[u*2]  /tempSize[u];
            B[u*2+1] = C[u*2+1]/tempSize[u];
            C[u*2] = 0;
            C[u*2+1] = 0;
            if (tempSize[u] != size[u]) {
                //printf("Changed %d %d\n",size[u],tempSize[u]);
                size[u] = tempSize[u];
                changed = 1;
            }
            tempSize[u] = 0;
        }
        for (int j = 0; j < K;j++){
            printf("%d | %f | %f -> %d\n",iterations,B[j*2],B[j*2+1],size[j]);
        }
        iterations++;
    }
}

int main (){
    alloc();
    inicializa();
    k_means();
}