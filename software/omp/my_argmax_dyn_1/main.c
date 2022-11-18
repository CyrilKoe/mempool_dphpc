
#include <inttypes.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "data.h"

#ifndef IS_MEMPOOL
#define IS_MEMPOOL 1
#endif

#if IS_MEMPOOL
#include "alloc.h"
#include "encoding.h"
#include "libgomp.h"
#include "printf.h"
#include "runtime.h"
#include "synchronization.h"
#else
#include "no_mempool.h"
#endif

typedef struct index_list {
  uint32_t idx;
  struct index_list *next;
} index_t; // sizeof(index_t) = 0x8

void free_all(alloc_t *tile_alloc, index_t *indexes) {
  index_t *tmp = indexes;
  while (tmp) {
    index_t *next_tmp = tmp->next;
    // printf("Free idx %i at addr %x\n", tmp->idx, tmp);
    domain_free(tile_alloc, tmp);
    tmp = next_tmp;
  }
}

void print_indexes(index_t *indexes) {
  index_t *tmp = indexes;
  while (tmp) {
    printf("%i ", tmp->idx);
    tmp = tmp->next;
  }
  printf("\n");
}

int main() {

#if IS_MEMPOOL
  uint32_t core_id = mempool_get_core_id();
  uint32_t num_cores = mempool_get_core_count();
  mempool_barrier_init(core_id);
  mempool_init(core_id, num_cores);
  if (core_id == 0) {
#endif

    int32_t global_max = -1;
    index_t *global_indexes = NULL;
    uint32_t global_indexes_len = 0;

#pragma omp parallel // num_threads(32)
    {
      uint32_t id = omp_get_thread_num();
      uint32_t tile_id = id / NUM_CORES_PER_TILE;
      uint32_t local_offset = 0;
      uint32_t local_data_len = l2_data_len / omp_get_num_threads();
      int32_t local_max = -1;
      index_t *local_indexes = NULL;
      uint32_t local_indexes_len = 0;
      alloc_t *tile_alloc = get_alloc_tile(tile_id);

      // Initialize local vector of data within the tile
      int32_t *local_vector = (int32_t *)domain_malloc(
          tile_alloc, local_data_len * sizeof(int32_t));
      if (!local_vector) {
        *local_vector = 1;
      }
      uint32_t local_i = 0;
#pragma omp for
      for (uint32_t i = 0; i < l2_data_len; ++i) {
        if (local_i == 0) {
          local_offset = i;
        }
        local_vector[local_i++] = l2_data_flat[i];
      }

      if (id == 0) {
        printf("All cores are ready to start\n");
      }
#pragma omp barrier
      mempool_start_benchmark();

      for (uint32_t i = 0; i < local_data_len; ++i) {
        // printf("id:%i, i:%i, j:%i, mat:%i, local_max:%i\n", id, i, j, a[i *
        // num_columns + j], local_max);
        if (local_vector[i] > local_max) {
          local_max = local_vector[i];
#pragma omp critical
          {
            free_all(tile_alloc, local_indexes);
            local_indexes =
                (index_t *)domain_malloc(tile_alloc, sizeof(index_t));
          }
          if (!local_indexes) {
            *(int32_t *)local_indexes = 1;
          }
          local_indexes->idx = local_offset + i;
          local_indexes->next = NULL;
          local_indexes_len = 1;
        } else if (local_vector[i] == local_max) {
          index_t *new_index;
#pragma omp critical
          { new_index = (index_t *)domain_malloc(tile_alloc, sizeof(index_t)); }
          if (!new_index) {
            *(int32_t *)new_index = 1;
          }
          new_index->next = local_indexes;
          new_index->idx = local_offset + i;
          ;
          local_indexes = new_index;
          local_indexes_len++;
          // printf("id:%i, adding addr %x, before addr %x\n", id,
          // local_indexes, local_indexes->next);
        }
      }

// Very important to avoid double free?
#pragma omp barrier

/* Critical section result starts */
#pragma omp critical
      {

        if (local_max > global_max) {
          // Delete previous result and add ours
          global_max = local_max;
          free_all(tile_alloc, global_indexes);
          global_indexes = local_indexes;
          global_indexes_len = local_indexes_len;
        } else if (local_max == global_max) {
          // Append the end of our result to the global
          index_t *tmp = local_indexes;
          while (tmp->next) {
            tmp = tmp->next;
          }
          tmp->next = global_indexes;
          global_indexes = local_indexes;
          global_indexes_len += local_indexes_len;
        } else {
          // Delete our result
          free_all(tile_alloc, local_indexes);
          local_indexes = NULL;
        }
      }
      /* Critical section result ends */
      mempool_stop_benchmark();
    }

    printf("Global max = %i\n", global_max);
    printf("Global indexes len = %u\n", global_indexes_len);
    index_t *tmp = global_indexes;
    while (tmp) {
      printf("-> %u ", tmp->idx);
      tmp = tmp->next;
    }
    printf("\n");

#if IS_MEMPOOL
  } else {
    while (1) {
      mempool_wfi();
      run_task(core_id);
    }
  }
#endif
}