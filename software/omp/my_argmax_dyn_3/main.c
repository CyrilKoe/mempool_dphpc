
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

uint32_t print_lock __attribute__((section(".l1")));
uint32_t malloc_lock __attribute__((section(".l1")));

int32_t all_local_max[NUM_BANKS]
    __attribute__((aligned(NUM_BANKS), section(".l1")));
index_t *all_local_indexes[NUM_BANKS]
    __attribute__((aligned(NUM_BANKS), section(".l1")));

// This vector contains all the tile locks, it is randomly located in l1 but who
// cares
uint32_t *locks[NUM_CORES / NUM_CORES_PER_TILE];

/* Logarithmic reduction */
uint32_t volatile red_barrier[NUM_BANKS]
    __attribute__((aligned(NUM_BANKS), section(".l1")));

void free_all(alloc_t *tile_alloc, index_t *indexes) {
  index_t *tmp = indexes;
  while (tmp) {
    index_t *next_tmp = tmp->next;
    // printf("Free idx %i at addr %x\n", tmp->idx, tmp);
    // printf("# Free %4x\n", tmp);
    domain_free(tile_alloc, tmp);
    tmp = next_tmp;
  }
}

void do_lock(uint32_t *lock) {
  uint32_t islocked;
  islocked = __atomic_fetch_or(lock, 1, __ATOMIC_SEQ_CST);
  while (islocked) {
#if IS_MEMPOOL
    mempool_wait(NUM_CORES_PER_TILE);
#else
    asm volatile("nop" ::);
    asm volatile("nop" ::);
    asm volatile("nop" ::);
    asm volatile("nop" ::);
#endif
    islocked = __atomic_fetch_or(lock, 1, __ATOMIC_SEQ_CST);
  }
}

void do_unlock(uint32_t *lock) {
  __atomic_fetch_and(lock, 0, __ATOMIC_SEQ_CST);
}

void mempool_log_reduction(uint32_t volatile step, uint32_t core_id,
                           uint32_t num_cores);

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

    print_lock = 0;
    malloc_lock = 0;

    // Initialize reduction barrier with 0 (avoid x)
    for (uint32_t i = 0; i < NUM_BANKS; i++) {
      red_barrier[i] = 0;
    }

#pragma omp parallel num_threads(NUM_CORES_BENCH)
    {
      // Utils
      uint32_t id = omp_get_thread_num();
      uint32_t tile_id = id / NUM_CORES_PER_TILE;
      uint32_t num_cores = omp_get_num_threads();
      alloc_t *tile_alloc = get_alloc_tile(tile_id);
      // This pointer (located in local stack) points the the tile_lock (located
      // somewhere in the tile)
      uint32_t *tile_lock = NULL;
      // Local data vector
      uint32_t local_offset = 0;
      uint32_t local_data_len = l2_data_len / num_cores;
      int32_t *local_vector = NULL;
      // Results
      uint32_t local_indexes_len = 0;
      all_local_max[id * BANKING_FACTOR] = DEFAULT_MAX_VALUE;
      all_local_indexes[id * BANKING_FACTOR] = NULL;

      if (id == 0)
        printf("Benchmark %u cores and %u datas (%u per core)\n",
               NUM_CORES_BENCH, l2_data_len, local_data_len);
    
    #pragma omp barrier

      // Initialize your local vector of data somewhere in the tile
      do_lock(&malloc_lock);
      local_vector = (int32_t *)domain_malloc(tile_alloc,
                                              local_data_len * sizeof(int32_t));
      if (!local_vector) {
        local_vector =
            (int32_t *)simple_malloc(local_data_len * sizeof(int32_t));
        if (!local_vector)
          printf("ERROR\n");
      }
      do_unlock(&malloc_lock);

      // Fill your local vector of data
      uint32_t local_i = 0;
#pragma omp for
      for (uint32_t i = 0; i < l2_data_len; ++i) {
        // Keep your offset for later
        if (local_i == 0)
          local_offset = i;

        local_vector[local_i++] = l2_data_flat[i];
      }

      // Start benchmark NOW
      if (id == 0)
        printf("All cores are ready to start\n");
#pragma omp barrier
      mempool_start_benchmark();

      // Initialize the tile lock somewhere in the tile
      if (id % NUM_CORES_PER_TILE == 0) {
        do_lock(&malloc_lock);
        locks[tile_id] =
            (uint32_t *)domain_malloc(tile_alloc, sizeof(uint32_t));
        if (!locks[tile_id]) {
          locks[tile_id] = (uint32_t *)simple_malloc(sizeof(index_t));
          if (!locks[tile_id])
            printf("ERROR\n");
        }
        do_unlock(&malloc_lock);
        *locks[tile_id] = 0;
      }

// Initialize your local pointer of the tile lock
#pragma omp barrier
      tile_lock = locks[tile_id];

      for (uint32_t i = 0; i < local_data_len; ++i) {

        /* There's a better maximum */
        if (local_vector[i] > all_local_max[id * BANKING_FACTOR]) {
          // Set your local max to him
          all_local_max[id * BANKING_FACTOR] = local_vector[i];

          // Free your local index list and alloc a new one
          do_lock(tile_lock);
          free_all(tile_alloc, all_local_indexes[id * BANKING_FACTOR]);
          all_local_indexes[id * BANKING_FACTOR] =
              (index_t *)domain_malloc(tile_alloc, sizeof(index_t));
          if (!all_local_indexes[id * BANKING_FACTOR]) {
            do_lock(&malloc_lock);
            all_local_indexes[id * BANKING_FACTOR] =
                (index_t *)simple_malloc(sizeof(index_t));
            if (!all_local_indexes[id * BANKING_FACTOR])
              printf("ERROR\n");
            do_unlock(&malloc_lock);
          }
          do_unlock(tile_lock);

          // Save this new max's index
          all_local_indexes[id * BANKING_FACTOR]->idx = local_offset + i;
          all_local_indexes[id * BANKING_FACTOR]->next = NULL;

          local_indexes_len = 1;

          /* Theres another time the same maximum */
        } else if (local_vector[i] == all_local_max[id * BANKING_FACTOR]) {

          // Allocate a new entry for its index
          index_t *new_index;
          do_lock(tile_lock);
          new_index = (index_t *)domain_malloc(tile_alloc, sizeof(index_t));
          if (!new_index) {
            do_lock(&malloc_lock);
            new_index = (index_t *)simple_malloc(sizeof(index_t));
            if (!new_index)
              printf("ERROR\n");
            do_unlock(&malloc_lock);
          }
          do_unlock(tile_lock);

          // Save this new max's index
          new_index->idx = local_offset + i;
          // Push this new max on the top of our result list
          new_index->next = all_local_indexes[id * BANKING_FACTOR];
          all_local_indexes[id * BANKING_FACTOR] = new_index;

          local_indexes_len++;
        }
      }

      // if (id == 0)
      // printf("Entering reduction\n");

      // No need for barrier here, its inside the log reduction
      mempool_log_reduction(2, id, num_cores);

      mempool_stop_benchmark();
    }

    // Get your results
    global_indexes = all_local_indexes[0];
    global_max = all_local_max[0];

    // Get the result size
    global_indexes_len = 0;
    index_t *tmp = global_indexes;
    while (tmp) {
      global_indexes_len++;
      tmp = tmp->next;
    }

    // Print all this
    printf("Global max = %i\n", global_max);
    printf("Global indexes len = %u\n", global_indexes_len);
    tmp = global_indexes;
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

void mempool_log_reduction(uint32_t volatile step, uint32_t core_id,
                           uint32_t num_cores) {
  // TODO : Comments
  uint32_t idx, step_idx = (step * (core_id / step)) * BANKING_FACTOR;
  uint32_t next_step, previous_step;
  register int32_t local_max;
  index_t *local_indexes = NULL;

  previous_step = step >> 1;

  // Check if the collegue arrived before
  if ((step - previous_step) ==
      __atomic_fetch_add(&red_barrier[step_idx + previous_step - 1],
                         previous_step, __ATOMIC_RELAXED)) {
    // He did, so compare your values
    local_max = DEFAULT_MAX_VALUE;
    idx = step_idx;
    // Overkill, there should be only two values to check
    while (idx < step_idx + step * BANKING_FACTOR) {

      if (all_local_max[idx] > local_max) {
        local_max = all_local_max[idx];
        local_indexes = all_local_indexes[idx];
      } else if (all_local_max[idx] == local_max) {
        index_t *tmp = all_local_indexes[idx];
        while (tmp->next) {
          tmp = tmp->next;
        }
        tmp->next = local_indexes;
        local_indexes = all_local_indexes[idx];
      }
      idx += previous_step * BANKING_FACTOR;
    }
    // Now you got the new local_max and local_indexes
    all_local_max[step_idx] = local_max;
    all_local_indexes[step_idx] = local_indexes;

    next_step = step << 1;
    __atomic_store_n(&red_barrier[step_idx + previous_step - 1], 0,
                     __ATOMIC_RELAXED);

    if (step == num_cores) {
      all_local_max[0] = all_local_max[step_idx];
      all_local_indexes[0] = all_local_indexes[step_idx];
      __sync_synchronize(); // Full memory barrier
#if IS_MEMPOOL
      wake_up_all();
      mempool_wfi();
#endif
    } else {
      mempool_log_reduction(next_step, core_id, num_cores);
    }

  } else {
#if IS_MEMPOOL
    mempool_wfi();
#endif
  }
}
