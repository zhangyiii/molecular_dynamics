#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "CL/opencl.h"
#include <time.h>
#include "parameters.h"
#ifdef ALTERA
    #include "AOCL_Utils.h"
    using namespace aocl_utils;
#endif
#ifdef NVIDIA
    #include "nvidia.h"
#endif

// OpenCL runtime configuration
cl_platform_id platform = NULL;
cl_device_id device;
cl_context context = NULL;
cl_command_queue queue;
cl_program program = NULL;
cl_kernel kernel;
cl_mem nearest_buf;
cl_mem output_energy_buf;
cl_mem output_force_buf;

// Problem data.
cl_float3 input_a[N] = {};
cl_float3 nearest[N] = {};
cl_float3 velocity[N] = {};

float output_energy[N] = {};
cl_float3 output_force[N] = {};
double kernel_total_time = 0.;

bool init_opencl();
void init_problem();
void run();
void cleanup();
void md();
void motion();
void nearest_image();
void calculate_energy_force_lj();

// Entry point.
int main() {
    time_t start_total_time = time(NULL);
    // Initialize OpenCL.
    if(!init_opencl()) {
      return -1;
    }
    // Initialize the problem data.
    init_problem();
    md();
    // Free the resources allocated
    cleanup();
    time_t end_total_time = time(NULL);
    printf("\nTotal execution time in seconds =  %f\n", difftime(end_total_time, start_total_time));
    printf("\nKernel execution time in milliseconds = %0.3f ms\n", (kernel_total_time / 1000000.0) );
    return 0;
}

/////// HELPER FUNCTIONS ///////

// Initializes the OpenCL objects.
bool init_opencl() {
    cl_int status;

    printf("Initializing OpenCL\n");
    #ifdef ALTERA
        if(!setCwdToExeDir()) {
          return false;
        }
        platform = findPlatform("Altera");
    #endif
    #ifdef NVIDIA
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
    #endif
    if(platform == NULL) {
      printf("ERROR: Unable to find OpenCL platform.\n");
      return false;
    }

    #ifdef ALTERA
        scoped_array<cl_device_id> devices;
        cl_uint num_devices;
        devices.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));
        // We'll just use the first device.
        device = devices[0];
    #endif
    #ifdef NVIDIA
        cl_uint num_devices;
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU , 1, &device, &num_devices);
    #endif

    // Create the context.
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
    checkError(status, "Failed to create context");

    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status, "Failed to create command queue");

    #ifdef ALTERA
        std::string binary_file = getBoardBinaryFile("md", device);
        printf("Using AOCX: %s\n", binary_file.c_str());
        program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);
    #endif
    #ifdef NVIDIA
        int MAX_SOURCE_SIZE  = 65536;
        FILE *fp;
        const char fileName[] = "./device/md.cl";
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
    #endif

    // Build the program that was just created.
    status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
    checkError(status, "Failed to build program");

    const char *kernel_name = "md";
    kernel = clCreateKernel(program, kernel_name, &status);
    checkError(status, "Failed to create kernel");

    // Input buffer.
    nearest_buf = clCreateBuffer(context, CL_MEM_READ_ONLY,
        N * sizeof(cl_float3), NULL, &status);
    checkError(status, "Failed to create buffer for input A");

    // Output buffer.
    output_energy_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        N * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for output_en");

     output_force_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        N * sizeof(cl_float3), NULL, &status);
    checkError(status, "Failed to create buffer for output_force");

    return true;
}

// Initialize the data for the problem. Requires num_devices to be known.
void init_problem() {
    int count = 0;
    for (double i = -(box_size - initial_dist_to_edge)/2; i < (box_size - initial_dist_to_edge)/2; i += initial_dist_by_one_axis) {
        for (double j = -(box_size - initial_dist_to_edge)/2; j < (box_size - initial_dist_to_edge)/2; j += initial_dist_by_one_axis) {
            for (double l = -(box_size - initial_dist_to_edge)/2; l < (box_size - initial_dist_to_edge)/2; l += initial_dist_by_one_axis) {
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
    nearest_image();
    for (int i = 0; i < N; i++){
        output_force[i] = (cl_float3){0, 0, 0};
        output_energy[i] = 0;
    }
    run();
}

void md() {
    for (int n = 0; n < total_it; n ++){
        calculate_energy_force_lj();
        motion();
        if (!(n % 500)){
            float total_energy = 0;
            for (int i = 0; i < N; i++)
                total_energy+=output_energy[i];
            total_energy/=(2 * N);
                printf("energy is %f \n",total_energy);
        }
    }
}

void run() {
    cl_int status;

    // Launch the problem for each device.
    cl_event kernel_event;
    cl_event finish_event;
    cl_ulong time_start, time_end;
    double total_time;
    // Each of the host buffers supplied to
    // clEnqueueWriteBuffer here is already aligned to ensure that DMA is used
    // for the host-to-device transfer.
    cl_event write_event;
    status = clEnqueueWriteBuffer(queue, nearest_buf, CL_FALSE,
        0, N * sizeof(cl_float3), nearest, 0, NULL, &write_event);
    checkError(status, "Failed to transfer input A");

    // Set kernel arguments.
    unsigned argi = 0;

    size_t global_work_size[1] = {N};
    size_t local_work_size[1] = {N};
    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &nearest_buf);
    checkError(status, "Failed to set argument input_a");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &output_energy_buf);
    checkError(status, "Failed to set argument output_en");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &output_force_buf);
    checkError(status, "Failed to set argument output_force");

    status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL,
        global_work_size, local_work_size, 1, &write_event, &kernel_event);
    checkError(status, "Failed to launch kernel");

    // Read the result. This the final operation.
    status = clEnqueueReadBuffer(queue, output_energy_buf, CL_FALSE,
        0, N * sizeof(float), output_energy, 1, &kernel_event, &finish_event);

    status = clEnqueueReadBuffer(queue, output_force_buf, CL_FALSE,
        0, N * sizeof(cl_float3), output_force, 1, &kernel_event, &finish_event);

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

void motion(){
    double total_energy = 0;
    for (int i = 0; i < N; i++)
            total_energy+=output_energy[i];
        total_energy/=(2 * N);
    for (int i = 0; i < N; i++) {
        velocity[i] = (cl_float3) {velocity[i].x + output_force[i].x * dt,
            velocity[i].y + output_force[i].y * dt,
            velocity[i].z + output_force[i].z * dt};
        input_a[i] = (cl_float3) {input_a[i].x + velocity[i].x * dt,
            input_a[i].y + velocity[i].y * dt,
            input_a[i].z + velocity[i].z * dt};
    }

}

void nearest_image(){
    for (int i = 0; i < N; i++){
        float x,y,z;
        if (input_a[i].x  > 0){
            x = fmod(input_a[i].x + half_box, box_size) - half_box;
        }
        else{
            x = fmod(input_a[i].x - half_box, box_size) + half_box;
        }
        if (input_a[i].y  > 0){
            y = fmod(input_a[i].y + half_box, box_size) - half_box;
        }
        else{
            y = fmod(input_a[i].y - half_box, box_size) + half_box;
        }
        if (input_a[i].z  > 0){
            z = fmod(input_a[i].z + half_box, box_size) - half_box;
        }
        else{
            z = fmod(input_a[i].z - half_box, box_size) + half_box;
        }
        nearest[i] = (cl_float3){ x, y, z};
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
    if(nearest_buf) {
      clReleaseMemObject(nearest_buf);
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

