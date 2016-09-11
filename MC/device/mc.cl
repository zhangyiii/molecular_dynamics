#define size 32
#define rc 3

__attribute__((reqd_work_group_size(size, size, 1)))
__kernel void mc(__global const float3 *restrict particles,
                 __global float *restrict out) {

    int index1 = get_global_id(0);
    int index2 = get_global_id(1);

    float sq_dist = (particles[index1].x - particles[index2].x)*(particles[index1].x - particles[index2].x) + (particles[index1].y - particles[index2].y)*(particles[index1].y - particles[index2].y) + (particles[index1].z - particles[index2].z)*(particles[index1].z - particles[index2].z);
    if ((sq_dist < rc * rc) && (index1 != index2)) {
        uint index = index1 * size + index2;
        float r6 = sq_dist * sq_dist * sq_dist;
        float r12 = r6 * r6;
        out[index] = 4 * (1 / r12 - 1 / r6);
    }
}

