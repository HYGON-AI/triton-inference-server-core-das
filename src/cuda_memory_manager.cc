// Copyright 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "cuda_memory_manager.h"

#ifdef USE_HCU
#include <string.h>

#include <set>
#include <sstream>
#else
#include <cnmem.h>
#include <string.h>

#include <set>
#endif

#include "cuda_utils.h"
#include "triton/common/logging.h"

namespace {

#ifndef USE_HCU
#define RETURN_IF_CNMEM_ERROR(S, MSG)                    \
  do {                                                   \
    auto status__ = (S);                                 \
    if (status__ != CNMEM_STATUS_SUCCESS) {              \
      return Status(                                     \
          Status::Code::INTERNAL,                        \
          (MSG) + ": " + cnmemGetErrorString(status__)); \
    }                                                    \
  } while (false)
#endif

std::string
PointerToString(void* ptr)
{
  std::stringstream ss;
  ss << ptr;
  return ss.str();
}

}  // namespace

namespace triton { namespace core {

std::unique_ptr<CudaMemoryManager> CudaMemoryManager::instance_;
std::mutex CudaMemoryManager::instance_mu_;

CudaMemoryManager::~CudaMemoryManager()
{
#ifdef USE_HCU
  // HCU：销毁或重置时显式同步，确保所有异步显存操作已完成
  hipDeviceSynchronize();
#else
  if (has_allocation_) {
    auto status = cnmemFinalize();
    if (status != CNMEM_STATUS_SUCCESS) {
      LOG_ERROR << "Failed to finalize CUDA memory manager: [" << status << "] "
                << cnmemGetErrorString(status);
    }
  }
#endif
}

void
CudaMemoryManager::Reset()
{
  std::lock_guard<std::mutex> lock(instance_mu_);
  instance_.reset();
}

Status
CudaMemoryManager::Create(const CudaMemoryManager::Options& options)
{
  // Ensure thread-safe creation of CUDA memory pool
  std::lock_guard<std::mutex> lock(instance_mu_);
  if (instance_ != nullptr) {
    LOG_WARNING << "New CUDA memory pools could not be created since they "
                   "already exists";
    return Status::Success;
  }

  std::set<int> supported_gpus;
  auto status = GetSupportedGPUs(
      &supported_gpus, options.min_supported_compute_capability_);
  if (status.IsOk()) {
#ifdef USE_HCU
    // HCU：使用驱动级异步内存池（hipMallocAsync/hipFreeAsync）替代 cnmem
    bool has_configured_device = false;

    for (auto gpu : supported_gpus) {
      const auto it = options.memory_pool_byte_size_.find(gpu);
      if ((it != options.memory_pool_byte_size_.end()) && (it->second != 0)) {
        // 检查当前设备是否支持驱动级 Memory Pool
        int is_pool_supported = 0;
        hipDeviceGetAttribute(
            &is_pool_supported, hipDeviceAttributeMemoryPoolsSupported, gpu);

        if (!is_pool_supported) {
          LOG_WARNING << "CUDA memory pool not supported on device " << gpu
                      << ", fallback will occur.";
          continue;
        }

        has_configured_device = true;
        hipMemPool_t mem_pool;
        RETURN_IF_CUDA_ERR(
            hipDeviceGetDefaultMemPool(&mem_pool, gpu),
            std::string("Failed to get default memory pool for GPU ") +
                std::to_string(gpu));

        // 设置内存池释放阈值，避免系统在 free 后频繁退回 OS
        uint64_t threshold = it->second;
        RETURN_IF_CUDA_ERR(
            hipMemPoolSetAttribute(
                mem_pool, hipMemPoolAttrReleaseThreshold, &threshold),
            std::string("Failed to set pool release threshold for GPU ") +
                std::to_string(gpu));

        LOG_INFO << "CUDA async memory pool configured on device " << gpu
                 << " with release threshold " << threshold;
      }
    }

    if (!has_configured_device) {
      LOG_INFO << "CUDA memory pool disabled or no supported devices found";
    }

    instance_.reset(new CudaMemoryManager(has_configured_device));
#else
    std::vector<cnmemDevice_t> devices;
    for (auto gpu : supported_gpus) {
      const auto it = options.memory_pool_byte_size_.find(gpu);
      if ((it != options.memory_pool_byte_size_.end()) && (it->second != 0)) {
        devices.emplace_back();
        auto& device = devices.back();
        memset(&device, 0, sizeof(device));
        device.device = gpu;
        device.size = it->second;

        LOG_INFO << "CUDA memory pool is created on device " << device.device
                 << " with size " << device.size;
      }
    }

    if (!devices.empty()) {
      RETURN_IF_CNMEM_ERROR(
          cnmemInit(devices.size(), devices.data(), CNMEM_FLAGS_CANNOT_GROW),
          std::string("Failed to finalize CUDA memory manager"));
    } else {
      LOG_INFO << "CUDA memory pool disabled";
    }

    // Use to finalize CNMeM properly when out of scope
    instance_.reset(new CudaMemoryManager(!devices.empty()));
#endif
  } else {
    return Status(
        status.ErrorCode(),
        "Failed to initialize CUDA memory manager: " + status.Message());
  }

  return Status::Success;
}

Status
CudaMemoryManager::Alloc(void** ptr, uint64_t size, int64_t device_id)
{
  if (instance_ == nullptr) {
    return Status(
        Status::Code::UNAVAILABLE, "CudaMemoryManager has not been created");
  } else if (!instance_->has_allocation_) {
    return Status(
        Status::Code::UNAVAILABLE,
#ifdef USE_HCU
        "CudaMemoryManager has no configured CUDA memory device");
#else
        "CudaMemoryManager has no preallocated CUDA memory");
#endif
  }

  int current_device;
#ifdef USE_HCU
  RETURN_IF_CUDA_ERR(
      hipGetDevice(&current_device), std::string("Failed to get device"));
#else
  RETURN_IF_CUDA_ERR(
      cudaGetDevice(&current_device), std::string("Failed to get device"));
#endif
  bool overridden = (current_device != device_id);
  if (overridden) {
#ifdef USE_HCU
    RETURN_IF_CUDA_ERR(
        hipSetDevice(device_id), std::string("Failed to set device"));
#else
    RETURN_IF_CUDA_ERR(
        cudaSetDevice(device_id), std::string("Failed to set device"));
#endif
  }

#ifdef USE_HCU
  // HCU：异步流式分配（默认流 0 自动维护依赖，零同步）
  auto err = hipMallocAsync(ptr, size, 0);
#else
  // Defer returning error to make sure the device is recovered
  auto err = cnmemMalloc(ptr, size, nullptr);
#endif

  if (overridden) {
#ifdef USE_HCU
    hipSetDevice(current_device);
#else
    cudaSetDevice(current_device);
#endif
  }

#ifdef USE_HCU
  RETURN_IF_CUDA_ERR(
#else
  RETURN_IF_CNMEM_ERROR(
#endif
      err, std::string("Failed to allocate CUDA memory with byte size ") +
               std::to_string(size) + " on GPU " + std::to_string(device_id));
  return Status::Success;
}

Status
CudaMemoryManager::Free(void* ptr, int64_t device_id)
{
  if (instance_ == nullptr) {
    return Status(
        Status::Code::UNAVAILABLE, "CudaMemoryManager has not been created");
  } else if (!instance_->has_allocation_) {
    return Status(
        Status::Code::UNAVAILABLE,
#ifdef USE_HCU
        "CudaMemoryManager has no configured CUDA memory device");
#else
        "CudaMemoryManager has no preallocated CUDA memory");
#endif
  }

  int current_device;
#ifdef USE_HCU
  RETURN_IF_CUDA_ERR(
      hipGetDevice(&current_device), std::string("Failed to get device"));
#else
  RETURN_IF_CUDA_ERR(
      cudaGetDevice(&current_device), std::string("Failed to get device"));
#endif
  bool overridden = (current_device != device_id);
  if (overridden) {
#ifdef USE_HCU
    RETURN_IF_CUDA_ERR(
        hipSetDevice(device_id), std::string("Failed to set device"));
#else
    RETURN_IF_CUDA_ERR(
        cudaSetDevice(device_id), std::string("Failed to set device"));
#endif
  }

#ifdef USE_HCU
  // HCU：异步释放显存回内存池，避免整卡同步
  auto err = hipFreeAsync(ptr, 0);
#else
  // Defer returning error to make sure the device is recovered
  auto err = cnmemFree(ptr, nullptr);
#endif

  if (overridden) {
#ifdef USE_HCU
    hipSetDevice(current_device);
#else
    cudaSetDevice(current_device);
#endif
  }

#ifdef USE_HCU
  RETURN_IF_CUDA_ERR(
#else
  RETURN_IF_CNMEM_ERROR(
#endif
      err, std::string("Failed to deallocate CUDA memory at address ") +
               PointerToString(ptr) + " on GPU " + std::to_string(device_id));
  return Status::Success;
}

}}  // namespace triton::core
