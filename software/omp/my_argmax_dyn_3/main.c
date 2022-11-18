
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

int32_t all_local_max[NUM_BANKS]
    __attribute__((aligned(NUM_BANKS), section(".l1")));
index_t *all_local_indexes[NUM_BANKS]
    __attribute__((aligned(NUM_BANKS), section(".l1")));

// This vector contains all the tile locks, it is randomly located in l1 but who
// cares
uint32_t *locks[NUM_CORES / NUM_CORES_PER_TILE];

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

void lock_tile(uint32_t *lock) {
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

void unlock_tile(uint32_t *lock) {
  __atomic_fetch_and(lock, 0, __ATOMIC_SEQ_CST);
}

void mempool_log_reduction(uint32_t volatile step, uint32_t core_id);

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

#pragma omp parallel num_threads(NUM_CORES)
    {
      // Utils
      uint32_t id = omp_get_thread_num();
      uint32_t tile_id = id / NUM_CORES_PER_TILE;
      alloc_t *tile_alloc = get_alloc_tile(tile_id);
      // This pointer (located in local stack) points the the tile_lock (located
      // somewhere in the tile)
      uint32_t *tile_lock = NULL;
      // Local data vector
      uint32_t local_offset = 0;
      uint32_t local_data_len = l2_data_len / omp_get_num_threads();
      // Results
      uint32_t local_indexes_len = 0;
      all_local_max[id * BANKING_FACTOR] = DEFAULT_MAX_VALUE;
      all_local_indexes[id * BANKING_FACTOR] = NULL;

      // Initialize the tile lock somewhere in the tile
      if (id % NUM_CORES_PER_TILE == 0) {
#pragma omp critical
        {
          locks[tile_id] =
              (uint32_t *)domain_malloc(tile_alloc, sizeof(uint32_t));
        }
        *locks[tile_id] = 0;
      }

// Initialize your local pointer of the tile lock
#pragma omp barrier
      tile_lock = locks[tile_id];

      // Initialize your local vector of data somewhere in the tile
      lock_tile(tile_lock);
      int32_t *local_vector = (int32_t *)domain_malloc(
          tile_alloc, local_data_len * sizeof(int32_t));
      unlock_tile(tile_lock);
      if (!local_vector)
        printf("ERROR\n");

      // Fill your local vector of data
      uint32_t local_i = 0;
#pragma omp for
      for (uint32_t i = 0; i < l2_data_len; ++i) {
        if (local_i == 0) {
          // Keep your offset for later
          local_offset = i;
        }
        local_vector[local_i++] = l2_data_flat[i];
      }

      // Start benchmark NOW
      if (id == 0) {
        printf("All cores are ready to start\n");
      }
#pragma omp barrier
      mempool_start_benchmark();

      for (uint32_t i = 0; i < local_data_len; ++i) {
        /* There's a better maximum */
        if (local_vector[i] > all_local_max[id * BANKING_FACTOR]) {
          // Set your local max to him
          all_local_max[id * BANKING_FACTOR] = local_vector[i];

          // Free your local index list and alloc a new one
          lock_tile(tile_lock);
          free_all(tile_alloc, all_local_indexes[id * BANKING_FACTOR]);
          all_local_indexes[id * BANKING_FACTOR] =
              (index_t *)domain_malloc(tile_alloc, sizeof(index_t));
          unlock_tile(tile_lock);

          if (!all_local_indexes[id * BANKING_FACTOR])
            printf("ERROR\n");

          // Save this new max's index
          all_local_indexes[id * BANKING_FACTOR]->idx = local_offset + i;
          all_local_indexes[id * BANKING_FACTOR]->next = NULL;

          local_indexes_len = 1;

          /* Theres another time the same maximum */
        } else if (local_vector[i] == all_local_max[id * BANKING_FACTOR]) {

          // Allocate a new entry for its index
          index_t *new_index;
          lock_tile(tile_lock);
          new_index = (index_t *)domain_malloc(tile_alloc, sizeof(index_t));
          unlock_tile(tile_lock);
          if (!new_index)
            printf("ERROR\n");

          // Save this new max's index
          new_index->idx = local_offset + i;
          // Push this new max on the top of our result list
          new_index->next = all_local_indexes[id * BANKING_FACTOR];
          all_local_indexes[id * BANKING_FACTOR] = new_index;

          local_indexes_len++;
        }
      }

      // No need for barrier here, its inside the log reduction
      mempool_log_reduction(2, id);

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

/* Logarithmic reduction */
uint32_t volatile red_barrier[NUM_BANKS]
    __attribute__((aligned(NUM_BANKS), section(".l1")));

void mempool_log_reduction(uint32_t volatile step, uint32_t core_id) {
  // TODO : Comments
  uint32_t idx, step_idx = (step * (core_id / step)) * BANKING_FACTOR;
  uint32_t next_step, previous_step;
  register int32_t local_max;
  index_t *local_indexes = NULL;

  previous_step = step >> 1;
  if ((step - previous_step) ==
      __atomic_fetch_add(&red_barrier[step_idx + previous_step - 1],
                         previous_step, __ATOMIC_RELAXED)) {

    local_max = DEFAULT_MAX_VALUE;
    idx = step_idx;
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
    all_local_max[step_idx] = local_max;
    all_local_indexes[step_idx] = local_indexes;

#if DEBUG
    lock_tile(&print_lock);
    if (all_local_max[step_idx] == 99) {
      printf("%u, %u, %i, %u, %4x", step, step_idx, all_local_max[step_idx],
             all_local_indexes[step_idx]->idx, all_local_indexes[step_idx]);
      index_t *tmp = all_local_indexes[step_idx];
      while (tmp) {
        printf(" -> %u", tmp->idx);
        tmp = tmp->next;
      }
      printf("\n");
    }
    unlock_tile(&print_lock);
#endif

    next_step = step << 1;
    __atomic_store_n(&red_barrier[step_idx + previous_step - 1], 0,
                     __ATOMIC_RELAXED);
    if (step == NUM_CORES) {
      all_local_max[0] = all_local_max[step_idx];
      all_local_indexes[0] = all_local_indexes[step_idx];
      __sync_synchronize(); // Full memory barrier
#if IS_MEMPOOL
      wake_up_all();
      mempool_wfi();
#endif
    } else {
      mempool_log_reduction(next_step, core_id);
    }

  } else {
#if IS_MEMPOOL
    mempool_wfi();
#endif
  }
}
