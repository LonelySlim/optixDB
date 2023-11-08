//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include <sampleConfig.h>

#include <sutil/Exception.h>
#include <sutil/sutil.h>

#include "state.h"
#include "timer.h"

#include <array>
#include <bitset>
#include <iomanip>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <map>
#include <vector>

#include <sutil/Camera.h>


template <typename T>
struct SbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<MissData> MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

typedef uint32_t CODE;
typedef uint32_t BITS;

//
//  variable
//
Timer                   timer_;
ScanState               state;

extern "C" void kGenAABB(double3 *points, double radius, unsigned int numPrims, OptixAabb *d_aabb, int epsilon);

static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */) {
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
              << message << "\n";
}

static void createVerticesArray(double3* vertices, std::ifstream& in)
{
    double p1,p2,p3;
    std::string line;
    int i = 0;
    while(std::getline(in,line))
    {
        std::istringstream iss(line);
        std::string element;
        std::getline(iss, element, ' ');
        p1 = std::stoi(element);
        std::getline(iss, element, ' ');
        p2 = std::stoi(element);
        std::getline(iss, element, ' ');
        p3 = std::stoi(element);
        vertices[i] = {p3,p2,p1};
        i++;
    }
    //std::cout << "the num of input vertices: " << i << std::endl; 
}

void initialize_optix(ScanState &state) {
    // Initialize CUDA
    CUDA_CHECK(cudaFree(0));

    // Initialize the OptiX API, loading all API entry points
    OPTIX_CHECK(optixInit());

    // Specify context options
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_cb;
    options.logCallbackLevel = 4;

    // Associate a CUDA context (and therefore a specific GPU) with this
    // device context
    CUcontext cuCtx = 0; // zero means take the current context
    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &state.context));
}

void make_gas(ScanState &state) {
    // Use default options for simplicity.  In a real use case we would want to
    // enable compaction, etc
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    const size_t vertices_size = sizeof(double3) * state.length;
    CUdeviceptr d_vertices = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_vertices), vertices_size));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(d_vertices),
        state.vertices,
        vertices_size,
        cudaMemcpyHostToDevice));
    state.params.points = reinterpret_cast<double3 *>(d_vertices);

    // set aabb
    OptixAabb *d_aabb;
    unsigned int numPrims = state.length;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_aabb), numPrims * sizeof(OptixAabb)));
    kGenAABB(reinterpret_cast<double3 *>(d_vertices), state.params.aabb_width / 2, numPrims, d_aabb, state.epsilon / 2);
    CUdeviceptr d_aabb_ptr = reinterpret_cast<CUdeviceptr>(d_aabb);

    // Our build input is a simple list of non-indexed triangle vertices
    const uint32_t vertex_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
    OptixBuildInput vertex_input = {};
    vertex_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    vertex_input.customPrimitiveArray.aabbBuffers = &d_aabb_ptr;
    vertex_input.customPrimitiveArray.flags = vertex_input_flags;
    vertex_input.customPrimitiveArray.numSbtRecords = 1;
    vertex_input.customPrimitiveArray.numPrimitives = numPrims;
    // it's important to pass 0 to sbtIndexOffsetBuffer
    vertex_input.customPrimitiveArray.sbtIndexOffsetBuffer = 0;
    vertex_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
    vertex_input.customPrimitiveArray.primitiveIndexOffset = 0;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        state.context,
        &accel_options,
        &vertex_input,
        1, // Number of build inputs
        &gas_buffer_sizes));
    CUdeviceptr d_temp_buffer_gas;
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&d_temp_buffer_gas),
        gas_buffer_sizes.tempSizeInBytes));

    CUDA_CHECK( cudaMalloc(
        reinterpret_cast<void**>(&state.d_gas_output_buffer),
        gas_buffer_sizes.outputSizeInBytes));

    OPTIX_CHECK(optixAccelBuild(
        state.context,
        0, // CUDA stream
        &accel_options,
        &vertex_input,
        1, // num build inputs
        d_temp_buffer_gas,
        gas_buffer_sizes.tempSizeInBytes,
        state.d_gas_output_buffer,
        gas_buffer_sizes.outputSizeInBytes,
        &state.gas_handle,
        nullptr, // emitted property list
        0              // num emitted properties
        ));

    // We can now free the scratch space buffer used during build and the vertex
    // inputs, since they are not needed by our trivial shading method
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_temp_buffer_gas)));
    // CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_vertices)));  // 放到了最后来释放 params.points 的空间
}

void make_module(ScanState &state) {
    char log[2048];

    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount     = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options.optLevel             = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel           = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

    state.pipeline_compile_options.usesMotionBlur = false;
    state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    state.pipeline_compile_options.numPayloadValues = 0;
    state.pipeline_compile_options.numAttributeValues = 0;
#ifdef DEBUG // Enables debug exceptions during optix launches. This may incur significant performance cost and should only be done during development.
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
    state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
    state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
    // By default (usesPrimitiveTypeFlags == 0) it supports custom and triangle primitives
    state.pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;

    const std::string ptx = sutil::getPtxString(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "optixDB.cu");
    size_t sizeof_log = sizeof(log);

    OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
        state.context,
        &module_compile_options,
        &state.pipeline_compile_options,
        ptx.c_str(),
        ptx.size(),
        log,
        &sizeof_log,
        &state.module));
}

void make_program_groups(ScanState &state) {
    char log[2048];

    OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros

    OptixProgramGroupDesc raygen_prog_group_desc = {};
    raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module = state.module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context,
        &raygen_prog_group_desc,
        1, // num program groups
        &program_group_options,
        log,
        &sizeof_log,
        &state.raygen_prog_group));

    OptixProgramGroupDesc miss_prog_group_desc = {};
    miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module = state.module;
    miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context,
        &miss_prog_group_desc,
        1, // num program groups
        &program_group_options,
        log,
        &sizeof_log,
        &state.miss_prog_group));

    OptixProgramGroupDesc hitgroup_prog_group_desc = {};
    hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_prog_group_desc.hitgroup.moduleIS = state.module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__cube";
      
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context,
        &hitgroup_prog_group_desc,
        1, // num program groups
        &program_group_options,
        log,
        &sizeof_log,
        &state.hitgroup_prog_group));
}

void make_pipeline(ScanState &state) {
    char log[2048];
    const uint32_t max_trace_depth = 1;
    std::vector<OptixProgramGroup> program_groups{state.raygen_prog_group, state.miss_prog_group, state.hitgroup_prog_group};

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = max_trace_depth;
    pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixPipelineCreate(
        state.context,
        &state.pipeline_compile_options,
        &pipeline_link_options,
        program_groups.data(),
        program_groups.size(),
        log,
        &sizeof_log,
        &state.pipeline));

    OptixStackSizes stack_sizes = {};
    for (auto &prog_group : program_groups) {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));
    }

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth,
                                           0, // maxCCDepth
                                           0, // maxDCDEpth
                                           &direct_callable_stack_size_from_traversal,
                                           &direct_callable_stack_size_from_state, &continuation_stack_size));
    OPTIX_CHECK(optixPipelineSetStackSize(state.pipeline, direct_callable_stack_size_from_traversal,
                                          direct_callable_stack_size_from_state, continuation_stack_size,
                                          1 // maxTraversableDepth
                                          ));
}

void make_sbt(ScanState &state) {
    CUdeviceptr raygen_record;
    const size_t raygen_record_size = sizeof(RayGenSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&raygen_record), raygen_record_size));
    RayGenSbtRecord rg_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(state.raygen_prog_group, &rg_sbt));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(raygen_record),
        &rg_sbt,
        raygen_record_size,
        cudaMemcpyHostToDevice));

    CUdeviceptr miss_record;
    size_t miss_record_size = sizeof(MissSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&miss_record), miss_record_size));
    MissSbtRecord ms_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(state.miss_prog_group, &ms_sbt));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(miss_record),
        &ms_sbt,
        miss_record_size,
        cudaMemcpyHostToDevice));

    CUdeviceptr hitgroup_record;
    size_t hitgroup_record_size = sizeof(HitGroupSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&hitgroup_record), hitgroup_record_size));
    HitGroupSbtRecord hg_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(state.hitgroup_prog_group, &hg_sbt));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(hitgroup_record),
        &hg_sbt,
        hitgroup_record_size,
        cudaMemcpyHostToDevice));

    state.sbt.raygenRecord = raygen_record;
    state.sbt.missRecordBase = miss_record;
    state.sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
    state.sbt.missRecordCount = 1;
    state.sbt.hitgroupRecordBase = hitgroup_record;
    state.sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
    state.sbt.hitgroupRecordCount = 1;
}

void cleanup(ScanState &state) {
    // free host memory

    // free device memory
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.sbt.raygenRecord)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.sbt.missRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.sbt.hitgroupRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.d_gas_output_buffer)));

    OPTIX_CHECK(optixPipelineDestroy(state.pipeline));
    OPTIX_CHECK(optixProgramGroupDestroy(state.hitgroup_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.miss_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.raygen_prog_group));
    OPTIX_CHECK(optixModuleDestroy(state.module));

    OPTIX_CHECK(optixDeviceContextDestroy(state.context));
}

void log_common_info(ScanState &state) {
    // printf("max ray num:                %dx%dx%d\n", state.width, state.height, state.depth);
    printf("data num:                   %d\n", state.length);
    printf("aabb_width                  %f\n", state.params.aabb_width);
    printf("ray_interval                %f\n", state.params.ray_interval);
    printf("epsilon: %d\n", state.epsilon);
}

void initializeOptix(std::ifstream& in, int length, 
                    uint32_t cube_width) {
    fprintf(stdout, "[OptiX]initializeOptix begin...\n");
    state.length = length;
    state.vertices = (double3 *) malloc(length * sizeof(double3));
    state.result_byte_num = ((state.length - 1) / 32 + 1) * 4;
    createVerticesArray(state.vertices,in);
    state.params.aabb_width = cube_width;
    state.params.ray_interval = state.params.aabb_width;
    state.epsilon = 0;
    
    // intersection_test_num, hit_num, predicate
    CUDA_CHECK(cudaMalloc(&state.params.predicate, 6 * sizeof(double)));
    
    log_common_info(state);

    timer_.commonGetStartTime(0); // record gas time
    initialize_optix(state);
    make_gas(state);
    make_module(state);
    make_program_groups(state);
    make_pipeline(state); // Link pipeline
    make_sbt(state);
    timer_.commonGetEndTime(0);
    timer_.showTime(0, "initializeOptix");
    fprintf(stdout, "[OptiX]initializeOptix end\n");
}

// direction = (x = 0, y = 1, z = 2)
// ray_mode = 0 for continuous ray, 1 for ray with space, 2 for ray as point
void refineWithOptix(BITS *dev_result_bitmap, double *predicate, 
                     int ray_length, bool inverse, int direction) {
    timer_.commonGetStartTime(1);
    
    double prange[3] = {
        predicate[1] - predicate[0],
        predicate[3] - predicate[2],
        predicate[5] - predicate[4]
    };
    
    double predicate_range;
    if (direction == 0) {
        predicate_range = prange[0];
        state.launch_width  = (int) (prange[1] / state.params.ray_interval) + 1;
        state.launch_height = (int) (prange[2] / state.params.ray_interval) + 1;
    } else if (direction == 1) {
        predicate_range = prange[1];
        state.launch_width  = (int) (prange[0] / state.params.ray_interval) + 1;
        state.launch_height = (int) (prange[2] / state.params.ray_interval) + 1;
    } else {
        predicate_range = prange[2];
        state.launch_width  = (int) (prange[0] / state.params.ray_interval) + 1;
        state.launch_height = (int) (prange[1] / state.params.ray_interval) + 1;
    }

   
    state.params.ray_space  = state.params.aabb_width;
    state.params.ray_length = ray_length;
    state.params.ray_stride = state.params.ray_length + state.params.ray_space;
    state.depth             = (int) (predicate_range / state.params.ray_stride);
    if (state.depth * state.params.ray_stride < predicate_range) {
        double last_stride = predicate_range - state.depth * state.params.ray_stride;
        state.params.ray_last_length = max(last_stride - state.params.ray_space, 0.0);
        if (state.params.ray_last_length == 0.0) {
            state.params.ray_last_length = 1e-5;
        }
        state.depth++;
    } else {
        state.params.ray_last_length = state.params.ray_length;
    }
    

    state.params.result = dev_result_bitmap;
    state.params.width  = state.launch_width;
    state.params.height = state.launch_height;
    state.params.direction = direction;
    state.params.handle = state.gas_handle;
    state.params.inverse = inverse;
    CUDA_CHECK(cudaMemcpy(state.params.predicate, predicate, 6 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.params.sum), state.launch_width * sizeof(int)));
    CUDA_CHECK(cudaMemset(state.params.sum, 0 , state.launch_width * sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.params.count), state.launch_width * sizeof(int)));
    CUDA_CHECK(cudaMemset(state.params.count, 0 , state.launch_width * sizeof(int)));
    
    //************
    //* Memcpy Params
    //************
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.d_params), sizeof(Params))); // Cannot malloc in initializeOptix phase
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(state.d_params),
        &state.params,
        sizeof(Params),
        cudaMemcpyHostToDevice));

    timer_.commonGetStartTime(2);
    OPTIX_CHECK(optixLaunch(state.pipeline, 0, state.d_params, sizeof(Params), &state.sbt, state.launch_width, state.launch_height, state.depth));
    CUDA_SYNC_CHECK();
    timer_.commonGetEndTime(2);
    timer_.commonGetEndTime(1);
    
    timer_.showTime(1, "refineWithOptix");
    timer_.showTime(2, "optixLaunch");
    timer_.clear();
    fprintf(stdout, "[OptiX] refineWithOptix end\n");
    
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(&state.params),
        reinterpret_cast<void *>(state.d_params),
        sizeof(Params),
        cudaMemcpyDeviceToHost));

    // cleanup
    CUDA_CHECK(cudaFree((void *)state.d_params));
}

int main(){
    //std::ifstream in("/home/sxr/rtdb/SDK/myOptixDB/outputdata.txt");
    std::ifstream in("/home/sxr/rtdb/SDK/optixDB/tools/generateData/uniform_data_100000000.0_10.txt");
    if(!in.is_open())
    {
        std::cerr << "can not open file outputdata.txt !" << std::endl;
        return 1;
    }
    initializeOptix(in,100000000,1);
    in.close();
    double predicate[6] = {0,101,1,10,0,100};
    refineWithOptix(nullptr,predicate,1,false,0);
    int *sum = (int *)malloc(10 * sizeof(int));
    int *count = (int *)malloc(10 * sizeof(int));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(sum),
        state.params.sum,
        10 * sizeof(int),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(count),
        state.params.count,
        10 * sizeof(int),
        cudaMemcpyDeviceToHost));
    for(int i = 0;i < 10;++i){
        if(count[i] != 0)
        std::cout << i  << " " << sum[i] << ' ' << count[i] <<  ' ' << ((float)sum[i]) / count[i] << std::endl;
    }
    cleanup(state);
    return 0;
}