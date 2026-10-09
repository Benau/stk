#define VMA_IMPLEMENTATION
#include "simd_wrapper.h"
#define VMA_SYSTEM_ALIGNED_MALLOC(size, alignment) \
    simd_aligned_alloc(alignment, size)
#define VMA_SYSTEM_ALIGNED_FREE(ptr) \
    simd_aligned_free(ptr)
#include "ge_vma.hpp"
