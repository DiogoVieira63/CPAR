#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <string.h>


#define MAX_IT 20

#define X(i) (i*2)
#define Y(i) (i*2+1)


int N , K, NumThreads;

float *vectorX;//[N] __attribute__((aligned (32)));
float *vectorY;//[N] __attribute__((aligned (32)));

float  *centroid;//,*vector;
int *size,*cluster;

//Function to allocate memory espace
void alloc() {
    //vector =  (float *) malloc(N*2*sizeof(float));
    vectorX = aligned_alloc(32,N * sizeof(float));
    vectorY = aligned_alloc(32,N * sizeof(float));

    cluster = (int *)   malloc(N *sizeof(int));
    centroid =(float *) malloc(K*2*sizeof(float));
    size = (int *) malloc(K*sizeof(int));
}

//Function to initialize the clusters and so the algorithm
void inicializa() {
    srand(10);
    
    // initialize vector with N values (x,y)
    for(int i = 0; i < N; i++) {
        vectorX[i] = (float) rand() / RAND_MAX;
        vectorY[i] = (float) rand() / RAND_MAX;
        //cluster[i] = 0;
    }
    // initialize cluster values
    #pragma omp parallel for num_threads(NumThreads)
    for(int i = 0; i < K; i++) {
        centroid[X(i)] = vectorX[i];
        centroid[Y(i)] = vectorY[i];
    }
}

//Function to calculate the euclidean distance
float distance (float x,float y, float *cl){
    return ((cl[0] - x) * (cl[0] - x)  + (cl[1] - y) * (cl[1] - y));
}

//Function to show the interactions after the execution time
void showResult(int it){
    printf("N = %d, K = %d\n",N,K);
    for (int i = 0;i<K ;i++){
        printf("Center: (%.3f,%.3f) : Size: %d\n",centroid[X(i)],centroid[Y(i)],size[i]);
    }
    printf("Iterations: %d\n",it);
}

//Function to assign every point to a cluster
void assignCluster(){
    #pragma omp parallel for num_threads(NumThreads)
    for (int i = 0; i < N ;i++){ 
        float lowest = distance(vectorX[(i)],vectorY[(i)],&centroid[X(0)]);
        int index_low = 0;
        for (int k = 1; k < K;k++){
            float dist = distance(vectorX[(i)],vectorY[(i)],&centroid[X(k)]);
                if (dist < lowest){
                    lowest = dist;
                    index_low = k;
                }
        }
        cluster[i]=index_low;
    }
}

/*
void assignClusterCollapse(){
    float lowest = 10;
    #pragma omp parallel for num_threads(NumThreads) collapse(2) firstprivate(lowest)
    for (int i = 0; i < N ;i++){ 
        for (int k = 0; k < K;k++){
            float dist = distance(vectorX[(i)],vectorY[(i)],&centroid[X(k)]);
            if (dist < lowest){
                lowest = dist;
                cluster[i] = k;
            }
            if(k == K-1) lowest =10;
        }
    }
}
*/

//Function to calculate centroids
void calculateCentroid(){
    int tempSize[K];
    float sumX[K];
    float sumY[K];
    memset( tempSize, 0,K*sizeof(int) );
    memset( sumX, 0,K*sizeof(float) );
    memset( sumY, 0,K*sizeof(float) );

    //Sum all values of x and y, and size of each cluster
    #pragma omp parallel for num_threads(NumThreads) reduction(+:sumX[:K],sumY[:K],tempSize[:K])
    for (int i = 0;i < N;i++){
        int index_low = cluster[i];
        sumX[(index_low)] += vectorX[(i)]; 
        sumY[(index_low)] += vectorY[(i)]; 
        tempSize[index_low]++;
    }
    //Calculate new values of each centroid
    #pragma omp parallel for num_threads(NumThreads)
    for (int i = 0; i < K; i++){
        size[i] = tempSize[i];
        centroid[X(i)] = sumX[(i)]/tempSize[i];
        centroid[Y(i)] = sumY[(i)]/tempSize[i];
    }
}

//Main Function of the program to run the k-means algorithm
int k_means(){ 
    for (int u = 0; u <= MAX_IT;u++){
        assignCluster();
        calculateCentroid();
    }
    return MAX_IT;
}

#define MIN(x, y) (((x) < (y)) ? (x) : (y))

int main (int argc, char ** argv){
    if (argc <= 4){
        N = atoi(argv[1]);
        K = atoi(argv[2]);
        NumThreads = argc == 3 ? 1 : MIN(atoi(argv[3]),omp_get_max_threads()); 

        alloc();
        inicializa();
        int iterations = k_means();
        showResult(iterations);
    }
    else {
        printf("Nº de argumentos inválido.");
    }

}