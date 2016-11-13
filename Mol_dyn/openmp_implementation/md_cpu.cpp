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
#define total_it 20000
#define dt 0.0005
#define initial_dist_by_one_axis 1.5

#define NUM_THREADS 4

struct dim {
    double x;
    double y;
    double z;
};
typedef struct dim dim;

double square_dist(dim first, dim second);
int nearest_image(dim* initial_array, dim* new_array, int particle_count, int particle);
void set_initial_state(dim *array, dim *velocity, dim *force);
double fast_pow(double a, int n);
void md(dim *array, dim *velocity, dim *force);
double calculate_energy_force_lj(dim *array, dim *force);
void motion(dim *array, dim *velocity, dim *force);

double Urc = 4 * ( 1 / fast_pow(rc, 12) - 1 / fast_pow(rc, 6) );

int main()
{
    time_t t;
    time_t start_total_time = time(NULL);
    srand((unsigned)time(&t));
    dim *r = (dim*)malloc(sizeof(dim) * N);
    dim *v = (dim*)malloc(sizeof(dim) * N);
    dim *f = (dim*)malloc(sizeof(dim) * N);
    set_initial_state(r,v,f);
    md(r,v,f);
    free(r);
    free(v);
    free(f);
    time_t end_total_time = time(NULL);
    printf("\nTotal execution time in seconds =  %f\n", difftime(end_total_time, start_total_time));
    return 0;
}

/////// HELPER FUNCTIONS ///////

void set_initial_state(dim *array, dim *velocity, dim *force) {
    int count = 0;
    for (double i = 0.5; i < box_size - 0.5; i += initial_dist_by_one_axis) {
        for (double j = 0.5; j < box_size - 0.5; j += initial_dist_by_one_axis) {
            for (double l = 0.5; l < box_size - 0.5; l += initial_dist_by_one_axis) {
                if( count == N){
                    return; //it is not balanced grid but we can use it
                }
                array[count] = { i,j,l };
                velocity[count] = { 0, 0, 0 };
                force[count] = { 0, 0, 0 };
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

double calculate_energy_force_lj(dim *array, dim *force){
    for (int i = 0; i < N; i++){
        force[i] = { 0, 0, 0};
    }
    double energy = 0;
    #pragma omp parallel for reduction(+:energy) num_threads(NUM_THREADS)
    for (int i = 0; i < N; i++) {
        double force_x = 0;
        double force_y = 0;
        double force_z = 0;
        dim *neighbors = (dim*)malloc(sizeof(dim) * N);
        int count_near = nearest_image(array, neighbors, N, i);
        for (int j = 0; j < count_near; j++) {
            double dist = square_dist(array[i], neighbors[j]);
            double r6 = fast_pow(dist, 3);
            double r12 = r6 * r6;
            double r8 = r6 * dist;
            double r14 = r12 * dist;
            double multiplier = (12 * (1 / r14 - 1 / r8));
            force_x += (neighbors[j].x - array[i].x)  * multiplier;
            force_y += (neighbors[j].y - array[i].y)  * multiplier;
            force_z += (neighbors[j].z - array[i].z)  * multiplier;
            energy += 4 * (1 / r12 - 1 / r6) - Urc;
        }
        free(neighbors);
        force[i].x = force_x;
        force[i].y = force_y;
        force[i].z = force_z;
    }
    // now we consider each interaction twice, so we need to divide energy by 2
    return energy / 2;
}

void md(dim *array, dim *velocity, dim *force) {
    for (int n = 0; n < total_it; n ++){
        double total_energy = calculate_energy_force_lj(array, force);
        motion(array, velocity, force);
        if (!(n % 1000)) {
            printf("energy is %f \n", total_energy/N);
        }
    }
}

void motion(dim *array, dim *velocity, dim * force){
    for (int i = 0; i < N; i++) {
        velocity[i] = {velocity[i].x + force[i].x * dt,
            velocity[i].y + force[i].y * dt,
            velocity[i].z + force[i].z * dt};
        array[i] = {array[i].x + velocity[i].x * dt,
            array[i].y + velocity[i].y * dt,
            array[i].z + velocity[i].z * dt};
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
