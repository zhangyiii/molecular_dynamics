Implementation of MD and MC algorithms on fpga, using altera's opencl compiler, compare results with GPGPU implementation

MC_one_to_all and MD_one_to_all- consider iteractions between specific particle and all others inside kernel
MC and MD - consider interactions betweeen pairs of particles inside kernel.

I suggest, that I'll obtain better performance on different potentials using different approach(one-to-one and one-to-all), so for now I maintain both methods inside one branch
openmp implementations located in MC and MD
