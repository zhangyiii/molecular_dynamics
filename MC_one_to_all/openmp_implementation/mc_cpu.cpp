#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <float.h>
#include <time.h>
#include <omp.h>
#include <string.h>

#define rc 3
#define box_size 6
#define N 32
#define nmax 60000
#define total_it 120000
#define T 1.3
#define initial_dist_by_one_axis 1.2

#define NUM_THREADS 8

struct dim {
    double x;
    double y;
    double z;
};
typedef struct dim dim;

double square_dist(dim first, dim second);
int nearest_image(dim* initial_array, dim* new_array, int particle_count, int particle);
void set_initial_state(dim *array);
double fast_pow(double a, int n);
void mc_method(dim *array);
double calculate_energy_lj(dim *array);

double Urc = 4 * ( 1 / fast_pow(rc, 12) - 1 / fast_pow(rc, 6) );
double max_deviation = 0.005;

int main()
{
    time_t t;
    time_t start_total_time = time(NULL);
    srand((unsigned)time(&t));
    dim *r = (dim*)malloc(sizeof(dim) * N);
    set_initial_state(r);
    mc_method(r);
    free(r);
    time_t end_total_time = time(NULL);
    printf("\nTotal execution time in seconds =  %f\n", difftime(end_total_time, start_total_time));
    return 0;
}

/////// HELPER FUNCTIONS ///////

void set_initial_state(dim *array) {
    int count = 0;
    for (double i = 1; i < box_size - 1; i += initial_dist_by_one_axis) {
        for (double j = 1; j < box_size - 1; j += initial_dist_by_one_axis) {
            for (double l = 1; l < box_size - 1; l += initial_dist_by_one_axis) {
                if( count == N){
                    return; //it is not balanced grid but we can use it
                }
                array[count] = { i,j,l };
                count++;
            }
        }
    }
    if( count < N ){
    	printf("error decrease initial_dist parameter, count is %ld  N is %ld \n", count, N);
    	exit(1);
    }
}

int nearest_image(dim* initial_array, dim* new_array, int particle_count, int particle) {
    int count = 0;
    for (int z_dif = -box_size; z_dif < box_size + 1; z_dif += box_size) {
        for (int y_dif = -box_size; y_dif < box_size + 1; y_dif += box_size) {
            for (int x_dif = -box_size; x_dif < box_size + 1; x_dif += box_size) {
                for (int i = 0; i < particle_count; i++) {
                    if (particle == i) {
                        continue;
                    }
                    dim temp = { initial_array[i].x + x_dif, initial_array[i].y + y_dif, initial_array[i].z + z_dif };
                    if (square_dist(temp, initial_array[particle]) < rc*rc) {
                        new_array[count] = temp;
                        count++;
                    }
                }
            }
        }
    }
    return count;
}

double square_dist(dim first, dim second) {
    return (first.x - second.x)*(first.x - second.x) + (first.y - second.y)*(first.y - second.y) + (first.z - second.z)*(first.z - second.z);
}

double calculate_energy_lj(dim *array){
    double energy = 0;
    #pragma omp parallel for reduction(+:energy) num_threads(NUM_THREADS)
    for (int i = 0; i < N; i++) {
        dim *neighbors = (dim*)malloc(sizeof(dim) * N);
        int count_near = nearest_image(array, neighbors, N, i);
        for (int j = 0; j < count_near; j++) {
            double dist = square_dist(array[i], neighbors[j]);
            double r6 = fast_pow(dist, 3);
            double r12 = r6 * r6;
            energy += 4 * (1 / r12 - 1 / r6) - Urc;
        }
        free(neighbors);
    }
    // now we consider each interaction twice, so we need to divide energy by 2
    return energy / 2;
}

void mc_method(dim *array) {
    double *energy_ar = (double*)malloc(sizeof(double) * nmax);
    register int i = 0;
    register int good_iter = 0;
    int good_iter_hung = 0;
    double u1 = calculate_energy_lj(array);
    while (1) {
        if ((good_iter == nmax) || (i == total_it)) {
            printf("\nenergy is %f \ngood iters percent %f \n", energy_ar[good_iter-1]/N, (float)good_iter/(float)total_it);
            break;
        }
        dim *tmp = (dim*)malloc(sizeof(dim)*N);
        memcpy(tmp, array, sizeof(dim)*N);
        for (int particle = 0; particle < N; particle++) {
            //ofsset between -max_deviation/2 and max_deviation/2
            double ex = (double)rand() / (double)RAND_MAX * max_deviation - max_deviation / 2;
            double ey = (double)rand() / (double)RAND_MAX * max_deviation - max_deviation / 2;
            double ez = (double)rand() / (double)RAND_MAX * max_deviation - max_deviation / 2;
            tmp[particle].x = tmp[particle].x + ex;
            tmp[particle].y = tmp[particle].y + ex;
            tmp[particle].z = tmp[particle].z + ex;
        }
        double u2 = calculate_energy_lj(tmp);
        double deltaU_div_T = (u1 - u2) / T;
        double probability = exp(deltaU_div_T);
        double rand_0_1 = (double)rand() / (double)RAND_MAX;
        if ((u2 < u1) || (probability <= rand_0_1)) {
            u1 = u2;
            memcpy(array, tmp, sizeof(dim)*N);
            energy_ar[good_iter] = u2;
            good_iter++;
            good_iter_hung++;
        }
        i++;
        free(tmp);
    }
}

inline double fast_pow(double a, int n) {
    if (n == 0)
        return 1;
    if (n % 2 == 1)
        return fast_pow(a, n - 1) * a;
    else {
        double b = fast_pow(a, n / 2);
        return b * b;
    }
}
