#include <string.h>
#include "CL/cl.h"
#define MAX_PLATFORMS_COUNT 2

void checkError(cl_int err, const char *operation){
    if (err != CL_SUCCESS){
        fprintf(stderr, "Error during operation '%s': %d\n", operation, err);
        exit(1);
    }
}