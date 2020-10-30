//Adapted from Udacity course 344

#ifndef __GPU_TIMER_H__
#define __GPU_TIMER_H__

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    void Start() {
        cudaEventRecord(start, 0);
    }
    
    void Stop() {
        cudaEventRecord(stop, 0);
    }
    
    float Elapsed() {
        float elapsed;
        
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};
   
#endif /* __GPU_TIMER_H__ */   
