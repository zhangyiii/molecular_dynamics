#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "CL/opencl.h"
#include "AOCL_Utils.h"

#define rc 3
#define box_size 6
#define N 32
#define nmax 20000
#define total_it 40000
#define T 1.3
#define initial_dist_by_one_axis 1.2

using namespace aocl_utils;

// OpenCL runtime configuration
cl_platform_id platform = NULL;
cl_device_id device;
cl_context context = NULL;
cl_command_queue queue;
cl_program program = NULL;
cl_kernel kernel;
cl_mem input_a_buf;
cl_mem output_buf;

// Problem data(positions and energy)
cl_float3 input_a[N] = {};
float output[N * N] = {};
double total_kernel_time = 0.;

// Function prototypes
bool init_opencl();
void init_problem();
void run();
void cleanup();
void mc();
float calculate_energy_lj();
float max_deviation = 0.005;

// Entry point.
int main() {
    const double start_time = getCurrentTimestamp();
    // Initialize OpenCL.
    if(!init_opencl()) {
      return -1;
    }
    // Initialize the problem data.
    init_problem();
    mc();
    // Free the resources allocated
    cleanup();
    const double end_time = getCurrentTimestamp();
    printf("\nTime: %0.3f ms\n", (end_time - start_time) * 1e3);
    printf("Kernel total time : %0.3f ms\n", double(total_kernel_time) * 1e-6);

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

    std::string binary_file = getBoardBinaryFile("mc", device);
    printf("Using AOCX: %s\n", binary_file.c_str());
    program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);

    // Build the program that was just created.
    status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
    checkError(status, "Failed to build program");

    const char *kernel_name = "mc";
    kernel = clCreateKernel(program, kernel_name, &status);
    checkError(status, "Failed to create kernel");

    // Input buffer.
    input_a_buf = clCreateBuffer(context, CL_MEM_READ_ONLY,
        N * sizeof(cl_float3), NULL, &status);
    checkError(status, "Failed to create buffer for input A");

    // Output buffer.
    output_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        N * N * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for output");

    return true;
}

// Initialize the data for the problem. Requires num_devices to be known.
void init_problem() {
    int count = 0;
    for (double i = 1.1; i < box_size - 1.1; i += initial_dist_by_one_axis) {
        for (double j = 1.1; j < box_size - 1.1; j += initial_dist_by_one_axis) {
            for (double l = 1.1; l < box_size - 1.1 ; l += initial_dist_by_one_axis) {
                if( count == N){
                    return; //it is not balanced grid but we can use it
                }
                input_a[count] = (cl_float3){ i, j, l };
                count++;
            }
        }
    }
    if( count < N ){
        printf("error decrease initial_dist parameter, count is %ld  N is %ld \n", count, N);
        exit(1);
    }
}

float calculate_energy_lj() {
    memset(output, 0, sizeof(output));
    run();
    float total_energy = 0;
    for (unsigned i = 0; i < N * N; i++)
        total_energy+=output[i];
    total_energy/=2;
    return total_energy;
}

void mc() {
    int i = 0;
    int good_iter = 0;
    int good_iter_hung = 0;
    float energy_ar[nmax] = {};
    float u1 = calculate_energy_lj();
    while (1) {
        if ((i % 100 == 0) && (i != 0)) {
            if ( good_iter_hung > 55 ){
                max_deviation  /=2;
            }
            if (good_iter_hung < 45 )
            {
                max_deviation *= 2;
            }
            good_iter_hung = 0;
        }
        if ((good_iter == nmax) || (i == total_it)) {
            printf("energy is %f \n", energy_ar[good_iter-1]/N);
            break;
        }
        cl_float3 tmp[N];
        memcpy(tmp, input_a, sizeof(cl_float3)*N);
        for (int particle = 0; particle < N; particle++) {
            //ofsset between -max_deviation/2 and max_deviation/2
            double ex = (double)rand() / (double)RAND_MAX * max_deviation - max_deviation / 2;
            double ey = (double)rand() / (double)RAND_MAX * max_deviation - max_deviation / 2;
            double ez = (double)rand() / (double)RAND_MAX * max_deviation - max_deviation / 2;
            input_a[particle].x = input_a[particle].x + ex;
            input_a[particle].y = input_a[particle].y + ex;
            input_a[particle].z = input_a[particle].z + ex;
        }
        double u2 = calculate_energy_lj();
        double deltaU_div_T = (u1 - u2) / T;
        double probability = exp(deltaU_div_T);
        double rand_0_1 = (double)rand() / (double)RAND_MAX;
        if ((u2 < u1) || (probability <= rand_0_1)) {
            u1 = u2;
            energy_ar[good_iter] = u2;
            good_iter++;
            good_iter_hung++;
        }
        else {
            memcpy(input_a, tmp, sizeof(cl_float3) * N);
        }
        i++;
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

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &output_buf);
    checkError(status, "Failed to set argument %d", argi - 1);

    status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL,
        global_work_size, local_work_size, 1, &write_event, &kernel_event);
    checkError(status, "Failed to launch kernel");

    // Read the result. This the final operation.
    status = clEnqueueReadBuffer(queue, output_buf, CL_FALSE,
        0, N * N * sizeof(float), output, 1, &kernel_event, &finish_event);

    // Release local events.
    clReleaseEvent(write_event);

    // Wait for all devices to finish.
    clWaitForEvents(1, &finish_event);

    cl_ulong time_ns = getStartEndTime(kernel_event);
    total_kernel_time += time_ns;

    // Release all events.
    clReleaseEvent(kernel_event);
    clReleaseEvent(finish_event);
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
    if(output_buf) {
      clReleaseMemObject(output_buf);
    }
    if(program) {
    clReleaseProgram(program);
    }
    if(context) {
    clReleaseContext(context);
    }
}