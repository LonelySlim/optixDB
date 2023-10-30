#ifndef OPTIXDB_H
#define OPTIXDB_H

struct Params
{
    double3*                points;             // d_pointer of vertices
    int                     width;              // number of ray
    int                     height;             // number of ray
    int                     direction;          // direction = (x = 0, y = 1, z = 2)
    double                  aabb_width;
    double                  ray_interval;
    double                  ray_space;
    double                  ray_length;         // length of each ray
    double                  ray_last_length;    // length of the last ray
    double                  ray_stride;         // ray_stride = ray_length + ray_space          
    double*                 predicate;
    OptixTraversableHandle  handle;
    unsigned int*           result;             // record scan result
    float*                  sum;
    int*                    count;
    double                  tmin;
    double                  tmax;
    bool                    inverse;
};


struct RayGenData
{
    // No data needed
};


struct MissData
{
};


struct HitGroupData
{
    double x1, x2;
    double y1, y2;
    double z1, z2;

    void print() {
        printf("predicate = {x1: %f, x2: %f, y1: %f, y2: %f, z1: %f, z2: %f}\n",
                x1, x2, y1, y2, z1, z2);
    }
};

typedef HitGroupData Predicate;

#endif