#define size 32
#define rc 3

__attribute__((reqd_work_group_size(size, size, 1)))
__kernel void md(__global const float3 *restrict particles,
                 __global float *restrict out_energy,
                 __global float3 *restrict out_force) {

    int index1 = get_global_id(0);
    int index2 = get_global_id(1);

    float3 r = (float3)((particles[index1].x - particles[index2].x), (particles[index1].y - particles[index2].y), (particles[index1].z - particles[index2].z));
    float sq_dist = r.x * r.x + r.y * r.y + r.z * r.z;
    if ((sq_dist < rc * rc) && (index1 != index2)) {
        uint index = index1 * size + index2;
        float r6 = sq_dist * sq_dist * sq_dist;
        float r12 = r6 * r6;
        float r8 = r6 * sq_dist;
        float r14 = r12 * sq_dist;
        float3 force = r * (12 * (1 / r14 - 1 / r8));
        float energy = 4 * (1 / r12 - 1 / r6);
        out_force[index] = force;
        out_energy[index] = energy;
    }
}

