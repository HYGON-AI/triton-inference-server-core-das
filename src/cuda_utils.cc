// Copyright 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cuda_utils.h"

#include "model_config_utils.h"
#include "triton/common/logging.h"
#include "triton/common/nvtx.h"

#ifdef TRITON_ENABLE_GPU
#ifdef USE_HCU
#include <hip/hip_runtime.h>
#else
#include <cuda.h>
#endif
#endif
#ifndef CUDART_CB
#define CUDART_CB
#endif

namespace triton { namespace core {

#ifdef TRITON_ENABLE_GPU
void CUDART_CB
MemcpyHost(void* args)
{
  auto* copy_params = reinterpret_cast<CopyParams*>(args);
  memcpy(copy_params->dst_, copy_params->src_, copy_params->byte_size_);
  delete copy_params;
}
#endif  // TRITON_ENABLE_GPU

Status
GetDeviceMemoryInfo(const int device_id, size_t* free, size_t* total)
{
  *free = 0;
  *total = 0;
#ifdef TRITON_ENABLE_GPU
  // Make sure that correct device is set before creating stream and
  // then restore the device to what was set by the caller.
#ifdef USE_HCU
  int current_device;
  auto cuerr = hipGetDevice(&current_device);
  bool overridden = false;
  if (cuerr == hipSuccess) {
    overridden = (current_device != device_id);
    if (overridden) {
      cuerr = hipSetDevice(device_id);
    }
  }

  if (cuerr == hipSuccess) {
    cuerr = hipMemGetInfo(free, total);
  }

  if (overridden) {
    hipSetDevice(current_device);
  }

  if (cuerr != hipSuccess) {
    return Status(
        Status::Code::INTERNAL,
        (std::string("unable to get memory info for device ") +
         std::to_string(device_id) + ": " + hipGetErrorString(cuerr)));
#else
  int current_device;
  auto cuerr = cudaGetDevice(&current_device);
  bool overridden = false;
  if (cuerr == cudaSuccess) {
    overridden = (current_device != device_id);
    if (overridden) {
      cuerr = cudaSetDevice(device_id);
    }
  }

  if (cuerr == cudaSuccess) {
    cuerr = cudaMemGetInfo(free, total);
  }

  if (overridden) {
    cudaSetDevice(current_device);
  }

  if (cuerr != cudaSuccess) {
    return Status(
        Status::Code::INTERNAL,
        (std::string("unable to get memory info for device ") +
         std::to_string(device_id) + ": " + cudaGetErrorString(cuerr)));
#endif
  }
#endif  // TRITON_ENABLE_GPU
  return Status::Success;
}

Status
EnablePeerAccess(const double min_compute_capability)
{
#ifdef TRITON_ENABLE_GPU
  // If we can't enable peer access for one device pair, the best we can
  // do is skipping it...
  std::set<int> supported_gpus;
  bool all_enabled = false;
  if (GetSupportedGPUs(&supported_gpus, min_compute_capability).IsOk()) {
    all_enabled = true;
    int can_access_peer = false;
    for (const auto& host : supported_gpus) {
#ifdef USE_HCU
      auto cuerr = hipSetDevice(host);

      if (cuerr == hipSuccess) {
        for (const auto& peer : supported_gpus) {
          if (host == peer) {
            continue;
          }

          cuerr = hipDeviceCanAccessPeer(&can_access_peer, host, peer);
          if ((cuerr == hipSuccess) && (can_access_peer == 1)) {
            cuerr = hipDeviceEnablePeerAccess(peer, 0);
          }

          all_enabled &= ((cuerr == hipSuccess) && (can_access_peer == 1));
#else
      auto cuerr = cudaSetDevice(host);

      if (cuerr == cudaSuccess) {
        for (const auto& peer : supported_gpus) {
          if (host == peer) {
            continue;
          }

          cuerr = cudaDeviceCanAccessPeer(&can_access_peer, host, peer);
          if ((cuerr == cudaSuccess) && (can_access_peer == 1)) {
            cuerr = cudaDeviceEnablePeerAccess(peer, 0);
          }

          all_enabled &= ((cuerr == cudaSuccess) && (can_access_peer == 1));
#endif
        }
      }
    }
  }
  if (!all_enabled) {
    return Status(
        Status::Code::UNSUPPORTED,
        "failed to enable peer access for some device pairs");
  }
#endif  // TRITON_ENABLE_GPU
  return Status::Success;
}

Status
CopyBuffer(
    const std::string& msg, const TRITONSERVER_MemoryType src_memory_type,
    const int64_t src_memory_type_id,
    const TRITONSERVER_MemoryType dst_memory_type,
    const int64_t dst_memory_type_id, const size_t byte_size, const void* src,
#ifdef USE_HCU
    void* dst, hipStream_t hip_stream, bool* cuda_used, bool copy_on_stream)
#else
    void* dst, cudaStream_t cuda_stream, bool* cuda_used, bool copy_on_stream)
#endif
{
  NVTX_RANGE(nvtx_, "CopyBuffer");

  *cuda_used = false;

  // For CUDA memcpy, all host to host copy will be blocked in respect to the
  // host, so use memcpy() directly. In this case, need to be careful on whether
  // the src buffer is valid.
  if ((src_memory_type != TRITONSERVER_MEMORY_GPU) &&
      (dst_memory_type != TRITONSERVER_MEMORY_GPU)) {
#ifdef TRITON_ENABLE_GPU
    if (copy_on_stream) {
      auto params = new CopyParams(dst, src, byte_size);
#ifdef USE_HCU
      hipLaunchHostFunc(
          hip_stream, MemcpyHost, reinterpret_cast<void*>(params));
#else
      cudaLaunchHostFunc(
          cuda_stream, MemcpyHost, reinterpret_cast<void*>(params));
#endif
      *cuda_used = true;
    } else {
      memcpy(dst, src, byte_size);
    }
#else
    memcpy(dst, src, byte_size);
#endif  // TRITON_ENABLE_GPU
  } else {
#ifdef TRITON_ENABLE_GPU
    RETURN_IF_CUDA_ERR(
#ifdef USE_HCU
        hipMemcpyAsync(dst, src, byte_size, hipMemcpyDefault, hip_stream),
#else
        cudaMemcpyAsync(dst, src, byte_size, cudaMemcpyDefault, cuda_stream),
#endif
        msg + ": failed to perform CUDA copy");

    *cuda_used = true;
#else
    return Status(
        Status::Code::INTERNAL,
        msg + ": try to use CUDA copy while GPU is not supported");
#endif  // TRITON_ENABLE_GPU
  }

  return Status::Success;
}

void
CopyBufferHandler(
    const std::string& msg, const TRITONSERVER_MemoryType src_memory_type,
    const int64_t src_memory_type_id,
    const TRITONSERVER_MemoryType dst_memory_type,
    const int64_t dst_memory_type_id, const size_t byte_size, const void* src,
#ifdef USE_HCU
    void* dst, hipStream_t hip_stream, void* response_ptr,
#else
    void* dst, cudaStream_t cuda_stream, void* response_ptr,
#endif
    triton::common::SyncQueue<std::tuple<Status, bool, void*>>*
        completion_queue)
{
  bool cuda_used = false;
  Status status = CopyBuffer(
      msg, src_memory_type, src_memory_type_id, dst_memory_type,
#ifdef USE_HCU
      dst_memory_type_id, byte_size, src, dst, hip_stream, &cuda_used);
#else
      dst_memory_type_id, byte_size, src, dst, cuda_stream, &cuda_used);
#endif
  completion_queue->Put(std::make_tuple(status, cuda_used, response_ptr));
}

#ifdef TRITON_ENABLE_GPU
Status
CheckGPUCompatibility(const int gpu_id, const double min_compute_capability)
{
  // Query the compute capability from the device
#ifdef USE_HCU
  hipDeviceProp_t cuprops;
  hipError_t cuerr = hipGetDeviceProperties(&cuprops, gpu_id);
  if (cuerr != hipSuccess) {
    return Status(
        Status::Code::INTERNAL,
        "unable to get CUDA device properties for GPU ID" +
            std::to_string(gpu_id) + ": " + hipGetErrorString(cuerr));
#else
  cudaDeviceProp cuprops;
  cudaError_t cuerr = cudaGetDeviceProperties(&cuprops, gpu_id);
  if (cuerr != cudaSuccess) {
    return Status(
        Status::Code::INTERNAL,
        "unable to get CUDA device properties for GPU ID" +
            std::to_string(gpu_id) + ": " + cudaGetErrorString(cuerr));
#endif
  }

  double compute_compability = cuprops.major + (cuprops.minor / 10.0);
  if ((compute_compability > min_compute_capability) ||
      (abs(compute_compability - min_compute_capability) < 0.01)) {
    return Status::Success;
  } else {
    return Status(
        Status::Code::UNSUPPORTED,
        "gpu " + std::to_string(gpu_id) + " has compute capability '" +
            std::to_string(cuprops.major) + "." +
            std::to_string(cuprops.minor) +
            "' which is less than the minimum supported of '" +
            std::to_string(min_compute_capability) + "'");
  }
}

Status
GetSupportedGPUs(
    std::set<int>* supported_gpus, const double min_compute_capability)
{
  // Make sure set is empty before starting
  supported_gpus->clear();

  int device_cnt;
#ifdef USE_HCU
  hipError_t cuerr = hipGetDeviceCount(&device_cnt);
  if ((cuerr == hipErrorNoDevice) || (cuerr == hipErrorInsufficientDriver)) {
    device_cnt = 0;
  } else if (cuerr != hipSuccess) {
    return Status(
        Status::Code::INTERNAL, "unable to get number of CUDA devices: " +
                                    std::string(hipGetErrorString(cuerr)));
#else
  cudaError_t cuerr = cudaGetDeviceCount(&device_cnt);
  if ((cuerr == cudaErrorNoDevice) || (cuerr == cudaErrorInsufficientDriver)) {
    device_cnt = 0;
  } else if (cuerr != cudaSuccess) {
    return Status(
        Status::Code::INTERNAL, "unable to get number of CUDA devices: " +
                                    std::string(cudaGetErrorString(cuerr)));
#endif
  }

  // populates supported_gpus
  for (int gpu_id = 0; gpu_id < device_cnt; gpu_id++) {
    Status status = CheckGPUCompatibility(gpu_id, min_compute_capability);
    if (status.IsOk()) {
      supported_gpus->insert(gpu_id);
    }
  }
  return Status::Success;
}

Status
SupportsIntegratedZeroCopy(const int gpu_id, bool* zero_copy_support)
{
  // Query the device to check if integrated
#ifdef USE_HCU
  hipDeviceProp_t cuprops;
  hipError_t cuerr = hipGetDeviceProperties(&cuprops, gpu_id);
  if (cuerr != hipSuccess) {
    return Status(
        Status::Code::INTERNAL,
        "unable to get CUDA device properties for GPU ID" +
            std::to_string(gpu_id) + ": " + hipGetErrorString(cuerr));
#else
  cudaDeviceProp cuprops;
  cudaError_t cuerr = cudaGetDeviceProperties(&cuprops, gpu_id);
  if (cuerr != cudaSuccess) {
    return Status(
        Status::Code::INTERNAL,
        "unable to get CUDA device properties for GPU ID" +
            std::to_string(gpu_id) + ": " + cudaGetErrorString(cuerr));
#endif
  }

  // Zero-copy supported only on integrated GPU when it can map host memory
  if (cuprops.integrated && cuprops.canMapHostMemory) {
    *zero_copy_support = true;
  } else {
    *zero_copy_support = false;
  }

  return Status::Success;
}

Status
GetAllocationGranularity(size_t& aligned_sz)
{
#ifdef USE_HCU
  hipMemAllocationProp prop = {};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = 0;

  RETURN_IF_ERROR(CudaDriverHelper::GetInstance().CuMemGetAllocationGranularity(
      &aligned_sz, &prop, hipMemAllocationGranularityMinimum));
#else
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = 0;

  RETURN_IF_ERROR(CudaDriverHelper::GetInstance().CuMemGetAllocationGranularity(
      &aligned_sz, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
#endif
  return Status::Success;
}

CudaDriverHelper::CudaDriverHelper()
{
  dl_open_handle_ = nullptr;
#ifndef _WIN32
#ifdef USE_HCU
  // HCU：优先加载 libamdhip64.so，兼容回退 libcuda.so（DTK shim）
  dl_open_handle_ = dlopen("libamdhip64.so", RTLD_LAZY);
  if (dl_open_handle_ == nullptr) {
    dl_open_handle_ = dlopen("libcuda.so", RTLD_LAZY);
  }
  if (dl_open_handle_ != nullptr) {
    void* cu_mem_create_fn = dlsym(dl_open_handle_, "hipMemCreate");
    if (cu_mem_create_fn == nullptr) {
      LOG_WARNING << "Failed to dlsym hipMemCreate";
      dl_open_handle_ = nullptr;
      return;
    }
    *((void**)&cu_mem_create_fn_) = cu_mem_create_fn;

    void* cu_get_error_string_fn = dlsym(dl_open_handle_, "hipDrvGetErrorString");
    if (cu_get_error_string_fn == nullptr) {
      LOG_WARNING << "Failed to dlsym hipDrvGetErrorString";
      dl_open_handle_ = nullptr;
      return;
    }
    *((void**)&cu_get_error_string_fn_) = cu_get_error_string_fn;

    void* cu_init_fn = dlsym(dl_open_handle_, "hipInit");
    if (cu_init_fn == nullptr) {
      LOG_WARNING << "Failed to dlsym hipInit";
      dl_open_handle_ = nullptr;
      return;
    }
    *((void**)&cu_init_fn_) = cu_init_fn;

    void* cu_mem_set_access_fn = dlsym(dl_open_handle_, "hipMemSetAccess");
    if (cu_mem_set_access_fn == nullptr) {
      LOG_WARNING << "Failed to dlsym hipMemSetAccess";
      dl_open_handle_ = nullptr;
      return;
    }
    *((void**)&cu_mem_set_access_fn_) = cu_mem_set_access_fn;

    void* cu_mem_release_fn = dlsym(dl_open_handle_, "hipMemRelease");
    if (cu_mem_release_fn == nullptr) {
      LOG_WARNING << "Failed to dlsym hipMemRelease";
      dl_open_handle_ = nullptr;
      return;
    }
    *((void**)&cu_mem_release_fn_) = cu_mem_release_fn;

    void* cu_mem_get_allocation_granularity_fn =
        dlsym(dl_open_handle_, "hipMemGetAllocationGranularity");
    if (cu_mem_get_allocation_granularity_fn == nullptr) {
      LOG_WARNING << "Failed to dlsym hipMemGetAllocationGranularity";
      dl_open_handle_ = nullptr;
      return;
    }
    *((void**)&cu_mem_get_allocation_granularity_fn_) =
        cu_mem_get_allocation_granularity_fn;

    void* cu_mem_address_free_fn = dlsym(dl_open_handle_, "hipMemAddressFree");
    if (cu_mem_address_free_fn == nullptr) {
      LOG_WARNING << "Failed to dlsym hipMemAddressFree";
      dl_open_handle_ = nullptr;
      return;
    }
    *((void**)&cu_mem_address_free_fn_) = cu_mem_address_free_fn;

    void* cu_mem_unmap_fn = dlsym(dl_open_handle_, "hipMemUnmap");
    if (cu_mem_unmap_fn == nullptr) {
      LOG_WARNING << "Failed to dlsym hipMemUnmap";
      dl_open_handle_ = nullptr;
      return;
    }
    *((void**)&cu_mem_unmap_fn_) = cu_mem_unmap_fn;

    void* cu_mem_address_reserve_fn =
        dlsym(dl_open_handle_, "hipMemAddressReserve");
    if (cu_mem_address_reserve_fn == nullptr) {
      LOG_WARNING << "Failed to dlsym hipMemAddressReserve";
      dl_open_handle_ = nullptr;
      return;
    }
    *((void**)&cu_mem_address_reserve_fn_) = cu_mem_address_reserve_fn;

    void* cu_mem_map_fn = dlsym(dl_open_handle_, "hipMemMap");
    if (cu_mem_map_fn == nullptr) {
      LOG_WARNING << "Failed to dlsym hipMemMap";
      dl_open_handle_ = nullptr;
      return;
    }
    *((void**)&cu_mem_map_fn_) = cu_mem_map_fn;

    // Initialize the driver API.
    hipError_t cuda_err = (*cu_init_fn_)(0 /* flags */);
    if (cuda_err != hipSuccess) {
      const char* error_string;
      (*cu_get_error_string_fn_)(cuda_err, &error_string);
      error_str_ = std::string("failed to call hipInit: ") + error_string;
      dlclose(dl_open_handle_);
      dl_open_handle_ = nullptr;
    }
#else
  dl_open_handle_ = dlopen("libcuda.so", RTLD_LAZY);
  if (dl_open_handle_ != nullptr) {
    void* cu_mem_create_fn = dlsym(dl_open_handle_, "cuMemCreate");
    if (cu_mem_create_fn == nullptr) {
      LOG_WARNING << "Failed to dlsym cuMemCreate";
      dl_open_handle_ = nullptr;
      return;
    }
    *((void**)&cu_mem_create_fn_) = cu_mem_create_fn;

    void* cu_get_error_string_fn = dlsym(dl_open_handle_, "cuGetErrorString");
    if (cu_get_error_string_fn == nullptr) {
      LOG_WARNING << "Failed to dlsym cuGetErrorString";
      dl_open_handle_ = nullptr;
      return;
    }
    *((void**)&cu_get_error_string_fn_) = cu_get_error_string_fn;

    void* cu_init_fn = dlsym(dl_open_handle_, "cuInit");
    if (cu_init_fn == nullptr) {
      LOG_WARNING << "Failed to dlsym cuInit";
      dl_open_handle_ = nullptr;
      return;
    }
    *((void**)&cu_init_fn_) = cu_init_fn;

    void* cu_mem_set_access_fn = dlsym(dl_open_handle_, "cuMemSetAccess");
    if (cu_mem_set_access_fn == nullptr) {
      LOG_WARNING << "Failed to dlsym cuMemSetAccess";
      dl_open_handle_ = nullptr;
      return;
    }
    *((void**)&cu_mem_set_access_fn_) = cu_mem_set_access_fn;

    void* cu_mem_release_fn = dlsym(dl_open_handle_, "cuMemRelease");
    if (cu_mem_release_fn == nullptr) {
      LOG_WARNING << "Failed to dlsym cuMemRelease";
      dl_open_handle_ = nullptr;
      return;
    }
    *((void**)&cu_mem_release_fn_) = cu_mem_release_fn;

    void* cu_mem_get_allocation_granularity_fn =
        dlsym(dl_open_handle_, "cuMemGetAllocationGranularity");
    if (cu_mem_get_allocation_granularity_fn == nullptr) {
      LOG_WARNING << "Failed to dlsym cuMemGetAllocationGranularity";
      dl_open_handle_ = nullptr;
      return;
    }
    *((void**)&cu_mem_get_allocation_granularity_fn_) =
        cu_mem_get_allocation_granularity_fn;

    void* cu_mem_address_free_fn = dlsym(dl_open_handle_, "cuMemAddressFree");
    if (cu_mem_address_free_fn == nullptr) {
      LOG_WARNING << "Failed to dlsym cuMemAddressFree";
      dl_open_handle_ = nullptr;
      return;
    }
    *((void**)&cu_mem_address_free_fn_) = cu_mem_address_free_fn;

    void* cu_mem_unmap_fn = dlsym(dl_open_handle_, "cuMemUnmap");
    if (cu_mem_unmap_fn == nullptr) {
      LOG_WARNING << "Failed to dlsym cuMemUnmap";
      dl_open_handle_ = nullptr;
      return;
    }
    *((void**)&cu_mem_unmap_fn_) = cu_mem_unmap_fn;

    void* cu_mem_address_reserve_fn =
        dlsym(dl_open_handle_, "cuMemAddressReserve");
    if (cu_mem_address_reserve_fn == nullptr) {
      LOG_WARNING << "Failed to dlsym cuMemAddressReserve";
      dl_open_handle_ = nullptr;
      return;
    }
    *((void**)&cu_mem_address_reserve_fn_) = cu_mem_address_reserve_fn;

    void* cu_mem_map_fn = dlsym(dl_open_handle_, "cuMemMap");
    if (cu_mem_map_fn == nullptr) {
      LOG_WARNING << "Failed to dlsym cuMemMap";
      dl_open_handle_ = nullptr;
      return;
    }
    *((void**)&cu_mem_map_fn_) = cu_mem_map_fn;

    // Initialize the driver API.
    CUresult cuda_err = (*cu_init_fn_)(0 /* flags */);
    if (cuda_err != CUDA_SUCCESS) {
      const char* error_string;
      (*cu_get_error_string_fn_)(cuda_err, &error_string);
      error_str_ = std::string("failed to call cuInit: ") + error_string;
      dlclose(dl_open_handle_);
      dl_open_handle_ = nullptr;
    }
#endif
  }
#endif
}

bool
CudaDriverHelper::IsAvailable()
{
  return dl_open_handle_ != nullptr;
}

Status
#ifdef USE_HCU
CudaDriverHelper::CuMemGetAllocationGranularity(
    size_t* aligned_size, const hipMemAllocationProp* prop,
    hipMemAllocationGranularity_flags flags)
#else
CudaDriverHelper::CuMemGetAllocationGranularity(
    size_t* aligned_size, const CUmemAllocationProp* prop,
    CUmemAllocationGranularity_flags flags)
#endif
{
  if (!IsAvailable()) {
    return Status(
        Status::Code::INTERNAL, "CudaDriverHelper has not been initialized.");
  }
  RETURN_IF_CUDA_DRIVER_ERR(
      cu_mem_get_allocation_granularity_fn_(aligned_size, prop, flags),
#ifdef USE_HCU
      std::string("failed to call hipMemGetAllocationGranularity"));
#else
      std::string("failed to call cuMemGetAllocationGranularity"));
#endif
  return Status::Success;
}

Status
#ifdef USE_HCU
CudaDriverHelper::CuMemCreate(
    hipMemGenericAllocationHandle_t* handle, size_t allocation_size,
    hipMemAllocationProp* prop, unsigned long long flags)
#else
CudaDriverHelper::CuMemCreate(
    CUmemGenericAllocationHandle* handle, size_t allocation_size,
    CUmemAllocationProp* prop, unsigned long long flags)
#endif
{
  if (!IsAvailable()) {
    return Status(
        Status::Code::INTERNAL, "CudaDriverHelper has not been initialized.");
  }
  RETURN_IF_CUDA_DRIVER_ERR(
      cu_mem_create_fn_(handle, allocation_size, prop, flags),
#ifdef USE_HCU
      std::string("failed to call hipMemCreate"));
#else
      std::string("failed to call cuMemCreate"));
#endif
  return Status::Success;
}

Status
#ifdef USE_HCU
CudaDriverHelper::CuMemSetAccess(
    hipDeviceptr_t ptr, size_t size, const hipMemAccessDesc* desc, size_t count)
#else
CudaDriverHelper::CuMemSetAccess(
    CUdeviceptr ptr, size_t size, const CUmemAccessDesc* desc, size_t count)
#endif
{
  if (!IsAvailable()) {
    return Status(
        Status::Code::INTERNAL, "CudaDriverHelper has not been initialized.");
  }
  RETURN_IF_CUDA_DRIVER_ERR(
      cu_mem_set_access_fn_(ptr, size, desc, count),
#ifdef USE_HCU
      std::string("failed to call hipMemSetAccess"));
#else
      std::string("failed to call cuMemSetAccess"));
#endif
  return Status::Success;
}

Status
#ifdef USE_HCU
CudaDriverHelper::CuMemMap(
    hipDeviceptr_t ptr, size_t size, size_t offset,
    hipMemGenericAllocationHandle_t handle, unsigned long long flags)
#else
CudaDriverHelper::CuMemMap(
    CUdeviceptr ptr, size_t size, size_t offset,
    CUmemGenericAllocationHandle handle, unsigned long long flags)
#endif
{
  if (!IsAvailable()) {
    return Status(
        Status::Code::INTERNAL, "CudaDriverHelper has not been initialized.");
  }
  RETURN_IF_CUDA_DRIVER_ERR(
      cu_mem_map_fn_(ptr, size, offset, handle, flags),
#ifdef USE_HCU
      std::string("failed to call hipMemMap"));
#else
      std::string("failed to call cuMemMap"));
#endif
  return Status::Success;
}

Status
#ifdef USE_HCU
CudaDriverHelper::CuMemRelease(hipMemGenericAllocationHandle_t handle)
#else
CudaDriverHelper::CuMemRelease(CUmemGenericAllocationHandle handle)
#endif
{
  if (!IsAvailable()) {
    return Status(
        Status::Code::INTERNAL, "CudaDriverHelper has not been initialized.");
  }
  RETURN_IF_CUDA_DRIVER_ERR(
#ifdef USE_HCU
      cu_mem_release_fn_(handle), std::string("failed to call hipMemRelease"));
#else
      cu_mem_release_fn_(handle), std::string("failed to call cuMemRelease"));
#endif
  return Status::Success;
}

void
#ifdef USE_HCU
CudaDriverHelper::CuGetErrorString(const char** error_string, hipError_t error)
#else
CudaDriverHelper::CuGetErrorString(const char** error_string, CUresult error)
#endif
{
  *error_string = nullptr;
  if (IsAvailable()) {
    cu_get_error_string_fn_(error, error_string);
  }
}

Status
#ifdef USE_HCU
CudaDriverHelper::CuMemAddressFree(hipDeviceptr_t ptr, size_t size)
#else
CudaDriverHelper::CuMemAddressFree(CUdeviceptr ptr, size_t size)
#endif
{
  if (!IsAvailable()) {
    return Status(
        Status::Code::INTERNAL, "CudaDriverHelper has not been initialized.");
  }
  RETURN_IF_CUDA_DRIVER_ERR(
#ifdef USE_HCU
      cu_mem_address_free_fn_(ptr, size),
      std::string("failed to call hipMemAddressFree"));
#else
      cu_mem_address_free_fn_(ptr, size),
      std::string("failed to call cuMemAddressFree"));
#endif
  return Status::Success;
}

Status
#ifdef USE_HCU
CudaDriverHelper::CuMemUnmap(hipDeviceptr_t ptr, size_t size)
#else
CudaDriverHelper::CuMemUnmap(CUdeviceptr ptr, size_t size)
#endif
{
  if (!IsAvailable()) {
    return Status(
        Status::Code::INTERNAL, "CudaDriverHelper has not been initialized.");
  }
  RETURN_IF_CUDA_DRIVER_ERR(
#ifdef USE_HCU
      cu_mem_unmap_fn_(ptr, size), std::string("failed to call hipMemUnmap"));
#else
      cu_mem_unmap_fn_(ptr, size), std::string("failed to call cuMemUnmap"));
#endif
  return Status::Success;
}

Status
#ifdef USE_HCU
CudaDriverHelper::CuMemAddressReserve(
    hipDeviceptr_t* ptr, size_t size, size_t alignment, hipDeviceptr_t addr,
    unsigned long long flags)
#else
CudaDriverHelper::CuMemAddressReserve(
    CUdeviceptr* ptr, size_t size, size_t alignment, CUdeviceptr addr,
    unsigned long long flags)
#endif
{
  if (!IsAvailable()) {
    return Status(
        Status::Code::INTERNAL, "CudaDriverHelper has not been initialized.");
  }
  RETURN_IF_CUDA_DRIVER_ERR(
#ifdef USE_HCU
      cu_mem_address_reserve_fn_(ptr, size, alignment, addr, flags),
      std::string("failed to call hipMemAddressReserve"));
#else
      cu_mem_address_reserve_fn_(ptr, size, alignment, addr, flags),
      std::string("failed to call cuMemAddressReserve"));
#endif
  return Status::Success;
}


CudaDriverHelper::~CudaDriverHelper()
{
#ifndef _WIN32
  if (dl_open_handle_ != nullptr) {
    dlclose(dl_open_handle_);
  }
#endif
}
#endif

}}  // namespace triton::core
