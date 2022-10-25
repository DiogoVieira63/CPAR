#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#define N 10000000

#define X(i) (i*2)
#define Y(i) (i*2+1)
#define K 4


float *vector, *centroid,*cluster;
int size[K] = {0};


void alloc() {
    vector =  (float *) malloc(N*2*sizeof(float));
    cluster = (float *) malloc(N *sizeof(float));
    centroid =(float *) malloc(K*2*sizeof(float));
    //sum =     (float *) malloc(K*2*sizeof(float));
}


void inicializa() {
    srand(10);
    // initialize vector with N values (x,y)
    for(int i = 0; i < N; i++) {
        vector[X(i)] = (float) rand() / RAND_MAX;
        vector[Y(i)] = (float) rand() / RAND_MAX;
        cluster[i] = 0;
    }
    // initialize cluster values
    for(int i = 0; i < K; i++) {
        centroid[X(i)]  = vector[X(i)];
        centroid[Y(i)]= vector[Y(i)];
    }
}



float distance1 (float x,float y, float *cl){
    return ((cl[0] - x) * (cl[0] - x)  + (cl[1] - y) * (cl[1] - y));
}

//float distance (float *vec, float *cl){
//    return sqrt(__builtin_powi(cl[0] - vec[0],2)  + __builtin_powi(cl[1] - vec[1],2));
//}
/*
void addToCluster(int numCluster, float x, float y){
    //cluster[numCluster][X(n)]= x;
    //cluster[numCluster][Y(n)]= y; 
    //sum[X(numCluster)] += x; 
    //sum[Y(numCluster)] += y; 
}
*/

void showResult(int it){
    printf("N = %d, K = %d\n",N,K);
    for (int i = 0;i<K ;i++){
        printf("Center: (%.3f,%.3f) : Size: %d\n",centroid[X(i)],centroid[Y(i)],size[i]);
    }
    printf("Iterations: %d\n",it);
}
/*
void calculate_centroids(){

    //for (int u = 0; u < K;u++){
    //    centroid[X(u)] = sum[X(u)]/size[u];
    //    centroid[Y(u)] = sum[Y(u)]/size[u];
    //    sum[X(u)] = 0;
    //    sum[Y(u)] = 0;
    //    size[u] = 0;
    //}
    float sum[K*2];
    for(int i =0; i < K;i++)sum[X(i)] = 0;
    for(int i = 0; i < N;i++){
        int index = cluster[i];
        sum[X(index)] += vector[X(i)];
        sum[Y(index)] += vector[Y(i)];
    }
    for (int i = 0; i < K; i++){
        centroid[X(i)] = sum[X(i)]/size[i];
        centroid[Y(i)] = sum[Y(i)]/size[i];
    }

}

int calculate_centroids2(){
    int changed = 0;
    for (int u = 0; u < K;u++){
        float sumX = 0, sumY =0, sizeCl = tempSize[u];
        for (int j = 0;j < sizeCl;j++){
            sumX += cluster[u][X(j)];
            sumY += cluster[u][Y(j)];
        }
        centroid[X(u)] = sumX/sizeCl;
        centroid[Y(u)] = sumY/sizeCl;
        //printf("%d\n",tempSize[u]);
        sum[X(u)] = 0;
        sum[Y(u)] = 0;
        if (tempSize[u] != size[u]) {
            size[u] = tempSize[u];
            changed = 1;
        }
        tempSize[u] = 0;
    }
    return changed;
}


void atrib_points(){
    int changed = 0;
    for (int i = 0; i < N ;i++){
        float lowest = 1.5f;
        int index_low = -1;
        for (int k = 0; k < K;k++){
            float dist = distance1(&vector[X(i)],&centroid[X(k)]);
            if (dist < lowest){
                lowest = dist;
                index_low = k;
            }
        }
        if (index_low != cluster[i]){
            cluster[i]=index_low;
            changed = 1;
        }
        size[index_low]++;
        //addToCluster(index_low,vector[X(i)],vector[Y(i)]);
    }
    return changed;
}
*/

void k_means(){
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
            //printf("%d | %d\n",index_low,cluster[i]);
            if (index_low != cluster[i]){
                cluster[i]=index_low;
                changed = 1;
            }
            sum[X(index_low)] += vector[X(i)]; 
            sum[Y(index_low)] += vector[Y(i)]; 
            tempSize[index_low]++;
        }
        for (int i = 0; i < K; i++){
            size[i] = tempSize[i];
            centroid[X(i)] = sum[X(i)]/size[i];
            centroid[Y(i)] = sum[Y(i)]/size[i];
        }


       //calculate_centroids();
       iterations++;
    }

    showResult(iterations - 1);
}

int main (){
    alloc();
    inicializa();
    k_means();
}