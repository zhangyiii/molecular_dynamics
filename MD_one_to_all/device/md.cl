#define size 32
#define rc 3

__attribute__((reqd_work_group_size(size, 1, 1)))
__kernel void md(__global const float3 *restrict particles,
                 __global float *restrict out_energy,
                 __global float3 *restrict out_force) {

    int index = get_global_id(0);
    float energy = 0;
    float3 force = 0;
    #pragma unroll 8
    for (int i = 0; i < size; i++) {
        float3 r = (float3)((particles[index].x - particles[i].x),
            (particles[index].y - particles[i].y),
            (particles[index].z - particles[i].z));
        float sq_dist = r.x * r.x + r.y * r.y + r.z * r.z;
        if ((sq_dist < rc * rc) && (i != index)) {
            float r6 = sq_dist * sq_dist * sq_dist;
            float r12 = r6 * r6;
            float r8 = r6 * sq_dist;
            float r14 = r12 * sq_dist;
            force += r * (12 * (1 / r14 - 1 / r8));
            energy += 4 * (1 / r12 - 1 / r6);
        }
    }
    out_force[index] = force;
    out_energy[index] = energy;
}

