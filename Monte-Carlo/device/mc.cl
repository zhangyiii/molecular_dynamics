#define size 128
#define rc 3

__attribute__((num_compute_units(1)))
__attribute__((reqd_work_group_size(size, 1, 1)))
__kernel void mc(__global const float3 *restrict particles,
                 __global float *restrict out) {

    int index = get_global_id(0);
    float energy = 0;
    #pragma unroll 8
    for (int i = 0; i < size; i++) {
        float sq_dist = (particles[i].x - particles[index].x)*(particles[i].x - particles[index].x)
             + (particles[i].y - particles[index].y)*(particles[i].y - particles[index].y)
             + (particles[i].z - particles[index].z)*(particles[i].z - particles[index].z);
        if ((sq_dist < rc * rc) && (i != index)) {
            float r6 = sq_dist * sq_dist * sq_dist;
            float r12 = r6 * r6;
            energy += 4 * (1 / r12 - 1 / r6);
        }
    }
    out[index] = energy;
}

