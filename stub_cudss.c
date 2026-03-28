/* stub_cudss.c — Minimal stub for libcudss.so.0 (CUDA Direct Sparse Solver)
 *
 * torch 2.8+ links against libcudss.so.0, but JetPack 6.2 CUDA 12.6 does
 * not ship this library. SNN forward-pass inference never calls sparse CUDA
 * solver ops, so empty symbol stubs satisfy the dynamic linker at import time.
 *
 * Symbols extracted via:
 *   strings libtorch_cuda.so | grep ^cudss
 */
void cudssCreate(void){}
void cudssDestroy(void){}
void cudssConfigCreate(void){}
void cudssConfigDestroy(void){}
void cudssDataCreate(void){}
void cudssDataDestroy(void){}
void cudssExecute(void){}
void cudssMatrixCreateCsr(void){}
void cudssMatrixCreateDn(void){}
void cudssMatrixDestroy(void){}
void cudssSetStream(void){}
