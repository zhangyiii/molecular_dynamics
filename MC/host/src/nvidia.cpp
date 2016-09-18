#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "CL/opencl.h"
#include "CL/cl.h"

#define MAX_PLATFORMS_COUNT 2

#define rc 3
#define box_size 6
#define N 32
#define nmax 20000
#define total_it 40000
#define T 1.3
#define initial_dist_by_one_axis 1.2

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
float max_deviation = 0.005;
double kernel_total_time = 0.;

// Function prototypes
bool init_opencl();
void init_problem();
void run();
void cleanup();
void mc();
float calculate_energy_lj();
void checkError(cl_int err, const char *operation);

// Entry point.
int main() {
    time_t start_total_time = time(NULL);
    // Initialize OpenCL.
    if(!init_opencl()) {
        return -1;
    }
    // Initialize the problem data.
    init_problem();
    mc();
    // Free the resources allocated
    cleanup();
    time_t end_total_time = time(NULL);
    printf("\nTotal execution time is %f\n", difftime(end_total_time, start_total_time));
    printf("\nKernel execution time in milliseconds = %0.3f ms\n", (kernel_total_time / 1000000.0) );
    return 0;
}

/////// HELPER FUNCTIONS ///////

// Initializes the OpenCL objects.(nvidia)
bool init_opencl() {
    cl_int status;

    printf("Initializing OpenCL\n");

    cl_uint num_platforms;
    cl_platform_id pls[MAX_PLATFORMS_COUNT];
    clGetPlatformIDs(MAX_PLATFORMS_COUNT, pls, &num_platforms);
    char vendor[128];
    for (int i = 0; i < MAX_PLATFORMS_COUNT; i++){
        clGetPlatformInfo (pls[i], CL_PLATFORM_VENDOR, sizeof(vendor), vendor, NULL);
        if (!strcmp("NVIDIA Corporation", vendor))
        {
            platform = pls[i];
            break;
        }
    }

    if(platform == NULL) {
      printf("ERROR: Unable to find Nvidia platform.\n");
      return false;
    }

    // Query the available OpenCL device.
    cl_uint num_devices;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU , 1, &device, &num_devices);

    // Create the context.
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
    checkError(status, "Failed to create context");

    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status, "Failed to create command queue");

    int MAX_SOURCE_SIZE  = 65536;
    FILE *fp;
    const char fileName[] = "./device/mc.cl";
    size_t source_size;
    char *source_str;
    try {
        fp = fopen(fileName, "r");
        if (!fp) {
            fprintf(stderr, "Failed to load kernel.\n");
            exit(1);
        }
        source_str = (char *)malloc(MAX_SOURCE_SIZE);
        source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
        fclose(fp);
    }
    catch (int a) {
        printf("%f", a);
    }
    program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &status);
    // Build the program that was just created.
    status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
    checkError(status, "Failed to build program");

    const char *kernel_name = "mc";
    kernel = clCreateKernel(program, kernel_name, &status);
    checkError(status, "Failed to create kernel");
    printf("create kernel");
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
    cl_ulong time_start, time_end;
    double total_time;
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
    checkError(status, "Failed to set argument input");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &output_buf);
    checkError(status, "Failed to set argument output");

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

    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    total_time = time_end - time_start;
    kernel_total_time += total_time;
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

void checkError(cl_int err, const char *operation)
{
    if (err != CL_SUCCESS){
        fprintf(stderr, "Error during operation '%s': %d\n", operation, err);
        exit(1);
    }
}