#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "CL/opencl.h"
#include "AOCL_Utils.h"

#define rc 3
#define box_size 7
#define N 32
#define total_it 20000
#define dt 0.0005
#define initial_dist_by_one_axis 1.5


using namespace aocl_utils;

// OpenCL runtime configuration
cl_platform_id platform = NULL;
cl_device_id device;
cl_context context = NULL;
cl_command_queue queue;
cl_program program = NULL;
cl_kernel kernel;
cl_mem input_a_buf;
cl_mem output_energy_buf;
cl_mem output_force_buf;

// Problem data.
cl_float3 input_a[N] = {};
cl_float3 velocity[N] = {};
//AFAIK it's imposible to pass 2d array to the kernel
float output_energy[N * N] = {};
cl_float3 output_force[N * N] = {};

// Function prototypes
bool init_opencl();
void init_problem();
void run();
void cleanup();
void md();
void motion();
void calculate_energy_force_lj();
// Entry point.
int main() {

    // Initialize OpenCL.
    if(!init_opencl()) {
      return -1;
    }

    // Initialize the problem data.
    init_problem();

    md();
    
    // Free the resources allocated
    cleanup();

  return 0;
}

/////// HELPER FUNCTIONS ///////

// Initializes the OpenCL objects.
bool init_opencl() {
    cl_int status;

    printf("Initializing OpenCL\n");

    if(!setCwdToExeDir()) {
      return false;
    }

    // Get the OpenCL platform.
    platform = findPlatform("Altera");
    if(platform == NULL) {
      printf("ERROR: Unable to find Altera OpenCL platform.\n");
      return false;
    }

    // Query the available OpenCL device.
    scoped_array<cl_device_id> devices;
    cl_uint num_devices;

    devices.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));

    // We'll just use the first device.
    device = devices[0];

    // Create the context.
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
    checkError(status, "Failed to create context");

    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status, "Failed to create command queue");

    std::string binary_file = getBoardBinaryFile("md", device);
    printf("Using AOCX: %s\n", binary_file.c_str());
    program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);

    // Build the program that was just created.
    status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
    checkError(status, "Failed to build program");

    const char *kernel_name = "md";
    kernel = clCreateKernel(program, kernel_name, &status);
    checkError(status, "Failed to create kernel");

    // Input buffer.
    input_a_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, 
        N * sizeof(cl_float3), NULL, &status);
    checkError(status, "Failed to create buffer for input A");

    // Output buffer.
    output_energy_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
        N * N * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for output");

     output_force_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
        N * N * sizeof(cl_float3), NULL, &status);
    checkError(status, "Failed to create buffer for output");

    return true;
}

// Initialize the data for the problem. Requires num_devices to be known.
void init_problem() {
    int count = 0;
    for (double i = 0.5; i < box_size - 0.5; i += initial_dist_by_one_axis) {
        for (double j = 0.5; j < box_size - 0.5; j += initial_dist_by_one_axis) {
            for (double l = 0.5; l < box_size - 0.5 ; l += initial_dist_by_one_axis) {
                if( count == N){
                    return; //it is not balanced grid but we can use it 
                }
                input_a[count] = (cl_float3){ i, j, l };
                velocity[count] = (cl_float3){ 0, 0, 0 };
                count++;
            }
        }
    }
    if( count < N ){
        printf("error decrease initial_dist parameter, count is %ld  N is %ld \n", count, N);
        exit(1);
    }
}

void calculate_energy_force_lj() {
    memset(output_energy, 0, sizeof(output_energy));
    for (int i = 0; i < N * N; i++)
        output_force[i] = (cl_float3){0, 0, 0};
    run();   
}

void md() {
    for (int n = 0; n < total_it; n ++){
        calculate_energy_force_lj();
        motion();
        float total_energy = 0;
        if (!(n % 500)) {
            for (unsigned i = 0; i < N * N; i++)
                total_energy+=output_energy[i];
            total_energy/=(2 * N);
            printf("energy is %f \n", total_energy);
        }
    }
}

void run() {
    cl_int status;

    // Launch the problem for each device.
    cl_event kernel_event;
    cl_event finish_event;

    // Each of the host buffers supplied to
    // clEnqueueWriteBuffer here is already aligned to ensure that DMA is used
    // for the host-to-device transfer.
    cl_event write_event;
    status = clEnqueueWriteBuffer(queue, input_a_buf, CL_FALSE,
        0, N * sizeof(cl_float3), input_a, 0, NULL, &write_event);
    checkError(status, "Failed to transfer input A");

    // Set kernel arguments.
    unsigned argi = 0;

    size_t global_work_size[2] = {32, 32};
    size_t local_work_size[2] = {32, 32};
    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &input_a_buf);
    checkError(status, "Failed to set argument %d", argi - 1);

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &output_energy_buf);
    checkError(status, "Failed to set argument %d", argi - 1);

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &output_force_buf);
    checkError(status, "Failed to set argument %d", argi - 1);

    status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL,
        global_work_size, local_work_size, 1, &write_event, &kernel_event);
    checkError(status, "Failed to launch kernel");

    // Read the result. This the final operation.
    status = clEnqueueReadBuffer(queue, output_energy_buf, CL_FALSE,
        0, N * N * sizeof(float), output_energy, 1, &kernel_event, &finish_event);

    status = clEnqueueReadBuffer(queue, output_force_buf, CL_FALSE,
        0, N * N * sizeof(cl_float3), output_force, 1, &kernel_event, &finish_event);

    // Release local events.
    clReleaseEvent(write_event);

    // Wait for all devices to finish.
    clWaitForEvents(1, &finish_event);

    // Release all events.
    clReleaseEvent(kernel_event);
    clReleaseEvent(finish_event);
}

void motion(){
    int count = 0;
    int particle_number = 0;
    cl_float3 force = (cl_float3) {0, 0, 0};
    for (int i = 0; i < N * N; i++) {
        if ((!(count % (N-1))) && (count)){
            velocity[particle_number] = (cl_float3) {velocity[particle_number].x + force.x * dt, 
                velocity[particle_number].y + force.y * dt, 
                velocity[particle_number].z + force.z * dt};
            input_a[particle_number] = (cl_float3) {input_a[particle_number].x + velocity[particle_number].x * dt,
                input_a[particle_number].y + velocity[particle_number].y * dt,
                input_a[particle_number].z + velocity[particle_number].z * dt};
            particle_number++;
            force = (cl_float3) {0, 0, 0};
            count = 0;
        }
        count++;
        force = (cl_float3) {output_force[i].x + force.x,
            output_force[i].y + force.y,
            output_force[i].z + force.z};
    }

}
// Free the resources allocated during initialization
void cleanup() {
    if(kernel) {
      clReleaseKernel(kernel);
    }
    if(queue) {
      clReleaseCommandQueue(queue);
    }
    if(input_a_buf) {
      clReleaseMemObject(input_a_buf);
    }
    if(output_energy_buf) {
      clReleaseMemObject(output_energy_buf);
    }
    if(output_force_buf) {
      clReleaseMemObject(output_force_buf);
    }
    if(program) {
    clReleaseProgram(program);
    }
    if(context) {
    clReleaseContext(context);
    }
}

