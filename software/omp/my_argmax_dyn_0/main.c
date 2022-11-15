
#include <inttypes.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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
#endif

void *my_malloc(size_t size) {
#if IS_MEMPOOL
  return simple_malloc(size);
#else
  return malloc(size);
#endif
}

void my_free(void *ptr) {
#if IS_MEMPOOL
  return simple_free(ptr);
#else
  return free(ptr);
#endif
}

typedef struct index_list {
  uint32_t idx;
  struct index_list *next;
} index_t;

void free_all(index_t *indexes) {
  index_t *tmp = indexes;
  while (tmp) {
    index_t *next_tmp = tmp->next;
    // printf("Free idx %i at addr %x\n", tmp->idx, tmp);
    my_free(tmp);
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
#else
  struct timespec tstart = {0, 0}, tend = {0, 0};
  clock_gettime(CLOCK_MONOTONIC, &tstart);
#endif

    int32_t global_max = -1;
    index_t *global_indexes = NULL;
    uint32_t global_indexes_len = 0;

#pragma omp parallel // num_threads(8)
    {
      uint32_t id = omp_get_thread_num();
      int32_t local_max = -1;
      index_t *local_indexes = NULL;
      uint32_t local_indexes_len = 0;

#pragma omp for
      for (uint32_t i = 0; i < l2_data_len; ++i) {
        // printf("id:%i, i:%i, j:%i, mat:%i, local_max:%i\n", id, i, j, a[i *
        // num_columns + j], local_max);
        if (l2_data_flat[i] > local_max) {
          local_max = l2_data_flat[i];
#pragma omp critical
          {
            free_all(local_indexes);
            local_indexes = (index_t *)my_malloc(sizeof(index_t));
          }
          local_indexes->idx = i;
          local_indexes->next = NULL;
          local_indexes_len = 1;
        } else if (l2_data_flat[i] == local_max) {
          index_t *new_index;
#pragma omp critical
          { new_index = (index_t *)my_malloc(sizeof(index_t)); }
          new_index->next = local_indexes;
          new_index->idx = i;
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
        printf("id:%i, local_max:%i, local_indexes_len:%u\n", id, local_max,
               local_indexes_len);

        if (local_max > global_max) {
          // Delete previous result and add ours
          global_max = local_max;
          free_all(global_indexes);
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
          free_all(local_indexes);
          local_indexes = NULL;
        }
      }
      /* Critical section result ends */
    }

    printf("Global max = %i\n", global_max);
    printf("Global indexes len = %u\n", global_indexes_len);

#if IS_MEMPOOL
  } else {
    while (1) {
      mempool_wfi();
      run_task(core_id);
    }
  }
#else
  clock_gettime(CLOCK_MONOTONIC, &tend);
  printf("Done : %f ns\n",
         ((double)tend.tv_sec + 1.0e-9 * tend.tv_nsec) -
             ((double)tstart.tv_sec + 1.0e-9 * tstart.tv_nsec));
#endif
}