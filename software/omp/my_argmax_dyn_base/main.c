
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
#define NUM_BANKS (NUM_CORES * BANKING_FACTOR)
#else
#include "no_mempool.h"
#define NUM_BANKS (NUM_CORES * BANKING_FACTOR)
#endif

#define DEBUG 0

#define DEFAULT_MAX_VALUE -(int32_t)(1 << 16);

typedef struct index_list {
  uint32_t idx;
  struct index_list *next;
} index_t; // sizeof(index_t) = 0x8

void free_all(index_t *indexes) {
  index_t *tmp = indexes;
  while (tmp) {
    index_t *next_tmp = tmp->next;
    // printf("Free idx %i at addr %x\n", tmp->idx, tmp);
    // printf("# Free %4x\n", tmp);
    simple_free(tmp);
    tmp = next_tmp;
  }
}

int32_t l1_data_flat[DATA_LEN]
    __attribute__((aligned(NUM_BANKS), section(".l1")));

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

    printf("Benchmark %u cores and %u datas (%u per core)\n", 1, l2_data_len,
           l2_data_len);

    // Fill your local vector of data
    for (uint32_t i = 0; i < l2_data_len; ++i) {
      l1_data_flat[i] = l2_data_flat[i];
    }

    printf("All cores are ready to start\n");
    mempool_start_benchmark();

    for (uint32_t i = 0; i < l2_data_len; ++i) {

      /* There's a better maximum */
      if (l1_data_flat[i] > global_max) {
        // Set your local max to him
        global_max = l1_data_flat[i];

        // Free your local index list and alloc a new one
        free_all(global_indexes);
        global_indexes = (index_t *)simple_malloc(sizeof(index_t));
        if (!global_indexes)
          printf("ERROR\n");

        // Save this new max's index
        global_indexes->idx = i;
        global_indexes->next = NULL;
        global_indexes_len = 1;

        /* Theres another time the same maximum */
      } else if (l1_data_flat[i] == global_max) {

        // Allocate a new entry for its index
        index_t *new_index;
        new_index = (index_t *)simple_malloc(sizeof(index_t));
        if (!new_index)
          printf("ERROR\n");

        // Save this new max's index
        new_index->idx = i;
        // Push this new max on the top of our result list
        new_index->next = global_indexes;
        global_indexes = new_index;
        global_indexes_len++;
      }
    }
    // Done
    mempool_stop_benchmark();

    // Print all this
    printf("Global max = %i\n", global_max);
    printf("Global indexes len = %u\n", global_indexes_len);
    index_t *tmp = global_indexes;
    while (tmp) {
      printf("-> %u ", tmp->idx);
      tmp = tmp->next;
    }
    printf("\n");

    // Todo save global_indexes and free all_global_indexes ?

#if IS_MEMPOOL
  } else {
    while (1) {
      mempool_wfi();
      run_task(core_id);
    }
  }
#endif
}