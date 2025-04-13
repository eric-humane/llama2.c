/**
 * FP16 utilities for ESP32-S3
 *
 * This file contains functions for handling half-precision (fp16) data
 * in the ESP32-S3 LLM implementation, allowing for on-the-fly conversion
 * to fp32 for better performance.
 */

#ifndef FP16_UTILS_H
#define FP16_UTILS_H

#include "esp_heap_caps.h"
#include "esp_log.h"
#include <stdint.h>

// Define IEEE 754 half-precision float format (16-bit)
typedef uint16_t float16_t;

/**
 * @brief Convert a single fp16 value to fp32
 *
 * @param h Input half-precision float value
 * @return float Output single-precision float value
 */
static inline float fp16_to_fp32(float16_t h) {
  // Extract components from fp16
  int sign = (h >> 15) & 0x1;
  int exp = (h >> 10) & 0x1F;
  int mant = h & 0x3FF;

  // Handle special cases
  if (exp == 0) {
    if (mant == 0) {
      // Zero
      return sign ? -0.0f : 0.0f;
    } else {
      // Denormalized number
      float val = sign ? -1.0f : 1.0f;
      val *= (float)mant / 1024.0f;
      val *= 1.0f / 16384.0f; // 2^-14
      return val;
    }
  } else if (exp == 31) {
    if (mant == 0) {
      // Infinity
      return sign ? -INFINITY : INFINITY;
    } else {
      // NaN
      return NAN;
    }
  }

  // Normalized number
  float val = sign ? -1.0f : 1.0f;
  val *= 1.0f + ((float)mant / 1024.0f);
  val *= powf(2.0f, (float)(exp - 15));

  return val;
}

/**
 * @brief Convert an array of fp16 values to fp32
 *
 * @param dst Destination buffer for fp32 values
 * @param src Source buffer with fp16 values
 * @param n Number of elements to convert
 * @param scale Optional scaling factor (nullptr for no scaling)
 */
static inline void fp16_to_fp32_array(float *dst, const float16_t *src, int n,
                                      const float *scale) {
  float scaling = scale ? *scale : 1.0f;

  for (int i = 0; i < n; i++) {
    dst[i] = fp16_to_fp32(src[i]) * scaling;
  }
}

/**
 * @brief Convert an array of fp16 values to fp32 with vector optimization
 *
 * @param dst Destination buffer for fp32 values (must be 16-byte aligned)
 * @param src Source buffer with fp16 values
 * @param n Number of elements to convert
 * @param scale Optional scaling factor
 */
void fp16_to_fp32_array_optimized(float *dst, const float16_t *src, int n,
                                  const float *scale);

/**
 * @brief Structure for handling fp16 model weights
 */
typedef struct {
  float16_t *fp16_weights; // Pointer to fp16 weights
  float *max_abs_values;   // Scaling factors for each weight tensor
  int num_tensor_groups;   // Number of tensor groups in the model
  int version;             // Model version
} FP16ModelData;

/**
 * @brief Initialize the FP16 model data structure
 *
 * @param model_data Pointer to the model data structure to initialize
 * @param weights_data Pointer to the raw weights data
 * @param num_tensors Number of tensor groups in the model
 * @param version Model version
 * @return int 0 on success, non-zero on failure
 */
int init_fp16_model_data(FP16ModelData *model_data, void *weights_data,
                         int num_tensors, int version);

/**
 * @brief Convert a tensor from fp16 to fp32 format
 *
 * @param dst Destination buffer for fp32 values
 * @param model_data FP16 model data structure
 * @param tensor_index Index of the tensor to convert
 * @param tensor_size Size of the tensor in elements
 * @return int 0 on success, non-zero on failure
 */
int convert_tensor_fp16_to_fp32(float *dst, const FP16ModelData *model_data,
                                int tensor_index, int tensor_size);

/**
 * @brief Free resources used by the FP16 model data
 *
 * @param model_data Pointer to the model data structure
 */
void free_fp16_model_data(FP16ModelData *model_data);

#endif /* FP16_UTILS_H */