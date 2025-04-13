/**
 * FP16 utilities for ESP32-S3 - Implementation
 */

#include "fp16_utils.h"
#include "esp_log.h"
#include "memory_utils.h"
#include "vector_simd.h"
#include <math.h>
#include <string.h>

static const char *TAG = "FP16_UTILS";

// Optimized fp16 to fp32 conversion using SIMD if available
void fp16_to_fp32_array_optimized(float *dst, const float16_t *src, int n,
                                  const float *scale) {
  float scaling = scale ? *scale : 1.0f;

  // For ESP32-S3, we can use DSP instructions for better performance
  // This is a simplified implementation; actual SIMD implementation would
  // require specific ESP32-S3 vector instructions

  // Process in chunks of 4 for potential SIMD optimization
  int i = 0;

  // Ensure both source and destination are properly aligned
  if (((uintptr_t)dst % 16 == 0) && ((uintptr_t)src % 16 == 0)) {
    // Aligned case - can use optimized conversion

    // Process 4 elements at a time
    for (; i <= n - 4; i += 4) {
      for (int j = 0; j < 4; j++) {
        dst[i + j] = fp16_to_fp32(src[i + j]) * scaling;
      }
    }
  }

  // Process remaining elements
  for (; i < n; i++) {
    dst[i] = fp16_to_fp32(src[i]) * scaling;
  }
}

int init_fp16_model_data(FP16ModelData *model_data, void *weights_data,
                         int num_tensors, int version) {
  if (!model_data || !weights_data || num_tensors <= 0) {
    ESP_LOGE(TAG, "Invalid parameters for init_fp16_model_data");
    return -1;
  }

  // Initialize the model data structure
  model_data->fp16_weights = (float16_t *)weights_data;
  model_data->num_tensor_groups = num_tensors;
  model_data->version = version;

  // Allocate memory for scaling factors
  model_data->max_abs_values =
      (float *)esp32_aligned_malloc(16, num_tensors * sizeof(float));

  if (!model_data->max_abs_values) {
    ESP_LOGE(TAG, "Failed to allocate memory for scaling factors");
    return -1;
  }

  // The max_abs_values array is stored at the end of the weights file
  // Calculate the offset where it starts
  size_t weights_size = 0; // This would be calculated based on model size

  // We need to know the total size of fp16 weights before we can locate the
  // scaling factors For now, we'll have to assume they are provided separately

  ESP_LOGI(TAG, "FP16 model data initialized with %d tensor groups",
           num_tensors);
  return 0;
}

int convert_tensor_fp16_to_fp32(float *dst, const FP16ModelData *model_data,
                                int tensor_index, int tensor_size) {
  if (!dst || !model_data || tensor_index < 0 ||
      tensor_index >= model_data->num_tensor_groups || tensor_size <= 0) {
    ESP_LOGE(TAG, "Invalid parameters for convert_tensor_fp16_to_fp32");
    return -1;
  }

  // Calculate the offset to the tensor in the fp16 weights
  size_t offset = 0;
  for (int i = 0; i < tensor_index; i++) {
    // In a real implementation, we'd need a way to know the size of each tensor
    // For simplicity, we're just assuming equal-sized tensors here
    offset += tensor_size;
  }

  // Get the scaling factor for this tensor
  float scale = model_data->max_abs_values[tensor_index];

  // Convert the tensor from fp16 to fp32
  float16_t *src = model_data->fp16_weights + offset;

  // Use optimized conversion if both buffers are properly aligned
  if (((uintptr_t)dst % 16 == 0) && ((uintptr_t)src % 16 == 0)) {
    fp16_to_fp32_array_optimized(dst, src, tensor_size, &scale);
  } else {
    // Fall back to standard conversion
    fp16_to_fp32_array(dst, src, tensor_size, &scale);
  }

  return 0;
}

void free_fp16_model_data(FP16ModelData *model_data) {
  if (model_data) {
    // Free the max_abs_values array
    if (model_data->max_abs_values) {
      esp32_aligned_free(model_data->max_abs_values);
      model_data->max_abs_values = NULL;
    }

    // We don't free fp16_weights as it's part of the loaded file
    model_data->fp16_weights = NULL;
    model_data->num_tensor_groups = 0;
  }
}