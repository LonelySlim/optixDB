#include <optix.h>

#include "optixDB.h"

#include <sutil/vec_math.h>

extern "C" {
__constant__ Params params;
}

static __forceinline__ __device__ void computeRay( uint3 idx, float3& origin, float3& direction ) {
    if (params.direction == 0) { // x
        origin = {
            float((params.predicate[0]) + idx.z * params.ray_stride - 1e-5),
            float(params.predicate[2] + (idx.x) * params.ray_interval),
            float(params.predicate[4] + (idx.y) * params.ray_interval)
        };
        direction = {1.0f, 0.0f, 0.0f};
    } else if (params.direction == 1) { // y
        origin = {
            float(params.predicate[0] + (idx.x) * params.ray_interval),
            float((params.predicate[2]) + idx.z * params.ray_stride),
            float(params.predicate[4] + (idx.y) * params.ray_interval),
        };
        direction = {0.0f, 1.0f, 0.0f};
    } else { // z
        origin = {
            float(params.predicate[0] + (idx.x) * params.ray_interval),
            float(params.predicate[2] + (idx.y) * params.ray_interval),
            float((params.predicate[4]) + idx.z * params.ray_stride)
        };
        direction = {0.0f, 0.0f, 1.0f};
    }
    //printf("%f %f %f %f\n",origin.x,origin.y,origin.z,direction.x);
}

static __forceinline__ __device__ void set_result(unsigned int *result, int idx) {
    int size = sizeof(unsigned int) << 3;
    int pos = idx / size;
    int pos_in_size = idx & (size - 1);
    // printf("set_result, idx: %d, pos_in_size: %d, 1 << (size - 1 - pos_in_size): %u\n", idx, pos_in_size, 1 << (size - 1 - pos_in_size));
    if (params.inverse) {
        atomicAnd( result + pos, ~(1 << (size - 1 - pos_in_size)) );
        // result[pos] &= ~(1 << (size - 1 - pos_in_size));
    } else {
        atomicOr( result + pos, 1 << (size - 1 - pos_in_size) );
        // result[pos] |= (1 << (size - 1 - pos_in_size));
    }
}

extern "C" __global__ void __raygen__rg() {
    // Lookup our location within the launch grid
    const uint3 idx = optixGetLaunchIndex();    
    const uint3 dim = optixGetLaunchDimensions(); 

    // Map our launch idx to a screen location and create a ray from the camera
    // location through the screen 
    float3 ray_origin, ray_direction;
    // computeRay_ray_interval_1( idx, ray_origin, ray_direction );
    computeRay( idx, ray_origin, ray_direction );

    // Trace the ray against our scene hierarchy
    double ray_length = params.ray_length + 1e-5 * 2;
    if (idx.z == dim.z - 1) {
        ray_length = params.ray_last_length + 1e-5 * 2;
    }
    //printf("%lf\n" ,ray_length);
    optixTrace(
            params.handle, 
            ray_origin,
            ray_direction,
            0.0f,                           // Min intersection distance
            (float) ray_length,      // Max intersection distance
            0.0f,                           // rayTime -- used for motion blur
            OptixVisibilityMask( 255 ),     // Specify always visible
            OPTIX_RAY_FLAG_NONE,
            0,                   // SBT offset   -- See SBT discussion
            1,                   // SBT stride   -- See SBT discussion
            0                    // missSBTIndex -- See SBT discussion
            );
}

extern "C" __global__ void __miss__ms()
{
    //do nothing
}


extern "C" __global__ void __closesthit__ch()
{
    //do nothing
}

extern "C" __global__ void __anyhit__ah()
{
    unsigned int primIdx = optixGetPrimitiveIndex();
    float3 ray_origin = optixGetObjectRayOrigin();
    float3 ray_direction = optixGetObjectRayDirection();
    const uint3 idx = optixGetLaunchIndex();
    float3 point = params.points[primIdx * 3];
    if(ray_direction.x != 0.0f){
        atomicAdd(params.count + idx.x, 1);
        atomicAdd(params.sum + idx.x, (int)point.z);
    } else if(ray_direction.y != 0.0f){
        //TODO:
    } else if(ray_direction.z != 0.0f){
        //TODO:
    }
    optixIgnoreIntersection();
}
