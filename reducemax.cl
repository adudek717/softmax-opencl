__kernel void reducemax(__global float* data, __local float* localData, __global float* outData)
{
    size_t globalId = get_global_id(0);
    size_t localSize = get_local_size(0);
    size_t localId = get_local_id(0);

    localData[localId] = data[globalId];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = localSize >> 1; i > 0; i >>= 1) {
        if (localId < i) {
            localData[localId] = max(localData[localId], localData[localId + i]); 
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(localId == 0) {
        outData[get_group_id(0)] = localData[0];
    }
}

__kernel void subtractmax(__global float* inData, __global float* outData, const float sub)
{
    size_t globalId = get_global_id(0);
    size_t globalSize = get_global_size(0);

    if (globalId < globalSize) {
        float subtractedVal = inData[globalId];
        subtractedVal = subtractedVal < 0.0f ? -subtractedVal : subtractedVal;
        outData[globalId] = exp(subtractedVal);
    }
}

__kernel void reducesum(__global float* data, __local float* localData, __global float* outData)
{
    size_t globalId = get_global_id(0);
    size_t localSize = get_local_size(0);
    size_t localId = get_local_id(0);

    localData[localId] = data[globalId];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = localSize >> 1; i > 0; i >>= 1) {
        if (localId < i) {
            localData[localId] += localData[localId + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(localId == 0) {
        outData[get_group_id(0)] = localData[0];
    }
}

__kernel void divideexpbysum(__global float* inData, __global float* outData, const float sum)
{
    size_t globalId = get_global_id(0);
    size_t globalSize = get_global_size(0);

    if (globalId < globalSize) {
        outData[globalId] = inData[globalId] / sum;
    }
}