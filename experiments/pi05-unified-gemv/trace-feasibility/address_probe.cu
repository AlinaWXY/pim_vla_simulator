// A tiny address-coverage check, not a workload/traffic model.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#define CUDA(call) do { auto e=(call); if(e!=cudaSuccess) { \
  std::fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e)); return 2; } } while(0)
__global__ void address_probe(const float* a,const float* b,float* c) {
  unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<64)c[i]=a[i]+b[(i*17)%64];
}
int main() {
  float a[64],b[64],c[64];
  for(int i=0;i<64;++i){a[i]=float(i);b[i]=float(64-i);}
  float *da,*db,*dc;
  CUDA(cudaMalloc(&da,sizeof(a)));CUDA(cudaMalloc(&db,sizeof(b)));CUDA(cudaMalloc(&dc,sizeof(c)));
  CUDA(cudaMemcpy(da,a,sizeof(a),cudaMemcpyHostToDevice));
  CUDA(cudaMemcpy(db,b,sizeof(b),cudaMemcpyHostToDevice));
  std::printf("PROBE_BASE a=%p b=%p c=%p bytes=256\n",(void*)da,(void*)db,(void*)dc);
  address_probe<<<2,32>>>(da,db,dc);
  CUDA(cudaGetLastError());CUDA(cudaDeviceSynchronize());
  CUDA(cudaMemcpy(c,dc,sizeof(c),cudaMemcpyDeviceToHost));
  double sum=0;
  for(int i=0;i<64;++i){if(c[i]!=a[i]+b[(i*17)%64])return 3;sum+=c[i];}
  std::printf("PROBE_PASS elements=64 sum=%.0f\n",sum);
  CUDA(cudaFree(da));CUDA(cudaFree(db));CUDA(cudaFree(dc));return 0;
}
