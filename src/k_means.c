#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#define N 10000000
//#define N 100

#define X(i) (i*2)
#define Y(i) (i*2+1)
#define K 4

float *vector, *centroid, *sum,**cluster;

int tempSize[K],size[K];

void alloc() {
    vector = (float *) malloc(N*2*sizeof(float));
    centroid = (float *) malloc(K*2*sizeof(float));
    sum = (float *) malloc(K*2*sizeof(float));


    cluster = (float **)malloc(K * sizeof(float *));
    for (int i = 0; i < K;i++){
        cluster[i] = (float *) malloc(N*2*sizeof(float));
    }
}


void inicializa() {
    srand(10);

    // initialize vector with N values (x,y)
    for(int i = 0; i < N; i++) {
        vector[X(i)] = (float) rand() / RAND_MAX;
        vector[Y(i)] = (float) rand() / RAND_MAX;
    }
    // initialize cluster values
    for(int i = 0; i < K; i++) {
        centroid[X(i)]  = vector[X(i)];
        centroid[Y(i)]= vector[Y(i)];
        size[i] = 0;
    }
}


float distance (float *vec, float *cl){
    return sqrt(pow(cl[0] - vec[0],2) + pow(cl[1] - vec[1],2));
}

void addToCluster(int numCluster, float x, float y){
    int n = tempSize[numCluster]++;
    cluster[numCluster][X(n)]= x;
    cluster[numCluster][Y(n)]= y; 
    sum[X(numCluster)] += x; 
    sum[Y(numCluster)] += y; 
}

void showResult(int it){

    printf("N = %d, K = %d\n",N,K);
    for (int i = 0;i<K ;i++){
        printf("Center: (%.3f,%.3f) : Size: %d\n",centroid[X(i)],centroid[Y(i)],size[i]);
    }
    printf("Iterations: %d\n",it);
}

void k_means(){
    int changed = 1, iterations = 0;
    while (changed){
        for (int i = 0; i < N ;i++){
            float lowest = distance(&vector[X(i)],centroid);
            int index_low = 0;
            for (int k = 1; k < K;k++){
                float dist = distance(&vector[X(i)],&centroid[X(k)]);
                if (dist < lowest){
                    lowest = dist;
                    index_low = k;
                }
            }
            addToCluster(index_low,vector[X(i)],vector[Y(i)]);
        }
        changed = 0;
        for (int u = 0; u < K;u++){
            centroid[X(u)] = sum[X(u)]/tempSize[u];
            centroid[Y(u)] = sum[Y(u)]/tempSize[u];
            sum[X(u)] = 0;
            sum[Y(u)] = 0;
            if (tempSize[u] != size[u]) {
                size[u] = tempSize[u];
                changed = 1;
            }
            tempSize[u] = 0;
        }
       iterations++;
    }
    showResult(iterations - 1);
}

int main (){
    alloc();
    inicializa();
    k_means();
}