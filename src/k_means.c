#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#define N 10000000

#define X(i) (i*2)
#define Y(i) (i*2+1)
#define K 4


float *vector, *centroid;
int size[K] = {0}, *cluster;

//Function to allocate memory espace
void alloc() {
    vector =  (float *) malloc(N*2*sizeof(float));
    cluster = (int *)   malloc(N *sizeof(int));
    centroid =(float *) malloc(K*2*sizeof(float));
}

//Function to initialize the clusters and so the algorithm
void inicializa() {
    srand(10);
    // initialize vector with N values (x,y)
    for(int i = 0; i < N; i++) {
        vector[X(i)] = (float) rand() / RAND_MAX;
        vector[Y(i)] = (float) rand() / RAND_MAX;
        //cluster[i] = 0;
    }
    // initialize cluster values
    for(int i = 0; i < K; i++) {
        centroid[X(i)]  = vector[X(i)];
        centroid[Y(i)]= vector[Y(i)];
    }
}

//Function to calculate the euclidean distance
float distance1 (float x,float y, float *cl){
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

//Main Function of the program to run the k-means algorithm
int k_means(){
    int changed = 1, iterations = 0;
    while (changed){
        changed = 0;
        int tempSize[K] = {0};
        float sum[K*2] = {0};
        for (int i = 0; i < N ;i++){ 
            float x  = vector[X(i)], y = vector[Y(i)];
            float lowest = distance1(x,y,&centroid[X(0)]);
            int index_low = 0;
            for (int k = 1; k < K;k++){
                float dist = distance1(x,y,&centroid[X(k)]);
                if (dist < lowest){
                    lowest = dist;
                    index_low = k;
                }
            }
            // Check if a value changed cluster
            if (index_low != cluster[i]){
                cluster[i]=index_low;
                changed = 1;
            }
            //sum values x,y for centroid calculation
            sum[X(index_low)] += vector[X(i)]; 
            sum[Y(index_low)] += vector[Y(i)]; 
            tempSize[index_low]++;
        }
        //centroid calculation

        for (int i = 0; i < K; i++){
            size[i] = tempSize[i];
            centroid[X(i)] = sum[X(i)]/size[i];
            centroid[Y(i)] = sum[Y(i)]/size[i];
        }
       iterations++;
    }
    return iterations-1;
}

int main (){
    alloc();
    inicializa();
    int iterations = k_means();
    showResult(iterations);

}