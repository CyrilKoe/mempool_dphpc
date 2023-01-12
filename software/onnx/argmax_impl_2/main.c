
#include <inttypes.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef IS_MEMPOOL
#define IS_MEMPOOL 1
#endif

// Include runtime
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

// Defines
#define DEBUG 0
#define DEFAULT_MAX_VALUE -(int32_t)(1 << 16);

// Linked list for argmax indexes
typedef struct index_list {
  uint32_t idx;
  struct index_list *next;
} index_t; // sizeof(index_t) = 0x8

// Take a lock
void do_lock(uint32_t *lock);

// Release a lock
void do_unlock(uint32_t *lock);

// Free a list of intermediate results
void free_all(alloc_t *tile_alloc, index_t *indexes);

// Argmax function
void argmax_int32(uint32_t data_len, int32_t *data, int32_t *global_max,
                  index_t **global_indexes);

// Logarithmic reduction
int32_t mempool_log_reduction(uint32_t volatile step, uint32_t core_id,
                              uint32_t num_cores);

////////////////////////////////////
/////// MAIN FUNCTION //////////////
////////////////////////////////////

// Include input data
#include "data.h"

int main() {

#if IS_MEMPOOL
  uint32_t core_id = mempool_get_core_id();
  uint32_t num_cores = mempool_get_core_count();
  mempool_barrier_init(core_id);
  mempool_init(core_id);
  if (core_id == 0) {
#endif

    // Global results
    int32_t global_max = -1;
    index_t *global_indexes = NULL;
    uint32_t global_indexes_len = 0;
    mempool_timer_t cycles;

    printf("Benchmark %u cores and %u datas (%u per core)\n", NUM_CORES_BENCH,
           l2_data_len, (uint32_t)l2_data_len / NUM_CORES_BENCH);
    printf("Expected global max = %u\n", expected_global_max);
    printf("Expected indexes len = %u\n", expected_indexes_len);

    // Call the kernel
    cycles = mempool_get_timer();
    argmax_int32(l2_data_len, l2_data_flat, &global_max, &global_indexes);
    cycles = mempool_get_timer() - cycles;

    // Get the result size
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
      // printf("-> %u ", tmp->idx);
      tmp = tmp->next;
    }
    printf("\n");
    printf("Duration: %d\n", cycles);

#if IS_MEMPOOL
  } else {
    while (1) {
      mempool_wfi();
      run_task(core_id);
    }
  }
#endif
}

////////////////////////////////////
/////// GLOBAL VARIABLES ///////////
////////////////////////////////////

// Global locks
uint32_t print_lock __attribute__((section(".l1")));
uint32_t malloc_lock __attribute__((section(".l1")));

// Global access to local intermediate results
int32_t all_local_max[NUM_BANKS]
    __attribute__((aligned(NUM_BANKS), section(".l1")));
index_t *all_local_indexes[NUM_BANKS]
    __attribute__((aligned(NUM_BANKS), section(".l1")));

// Global access to all the tile locks
uint32_t all_tiles_locks[NUM_BANKS]
    __attribute__((aligned(NUM_BANKS), section(".l1")));

// Logarithmic barrier lock
uint32_t volatile red_barrier[NUM_BANKS]
    __attribute__((aligned(NUM_BANKS), section(".l1")));

////////////////////////////////////
/////// UTILITY FUNCTIONS //////////
////////////////////////////////////

// Free a list of intermediate results
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

// Take a lock
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

// Release a lock
void do_unlock(uint32_t *lock) {
  __atomic_fetch_and(lock, 0, __ATOMIC_SEQ_CST);
}

////////////////////////////////////
/////// ARGMAX IMPLEMENTATION //////
////////////////////////////////////

void argmax_int32(uint32_t data_len, int32_t *data, int32_t *global_max,
                  index_t **global_indexes) {

#pragma omp parallel num_threads(NUM_CORES_BENCH)
  { mempool_start_benchmark(); }

  // Initialize global locks
  print_lock = 0;
  malloc_lock = 0;
  // Initialize global reduction barrier and tile locks
  for (uint32_t i = 0; i < NUM_BANKS; i++) {
    red_barrier[i] = 0;
    all_tiles_locks[i] = 0;
  }

#pragma omp parallel num_threads(NUM_CORES_BENCH)
  {
    // Utils
    uint32_t id = omp_get_thread_num();
    uint32_t tile_id = id / NUM_CORES_PER_TILE;
    uint32_t num_cores = omp_get_num_threads();
    // Local pointers to tile utils
    alloc_t *tile_alloc = get_alloc_tile(tile_id);
    uint32_t *tile_lock_ptr = NULL;
    // Local data vector
    uint32_t local_offset = 0;
    uint32_t local_data_len = data_len / num_cores;
    int32_t *local_vector = NULL;
    // Initialize local intermediary results
    uint32_t local_indexes_len = 0;
    index_t *local_indexes = NULL;
    int32_t local_max = DEFAULT_MAX_VALUE;
    // Initialize global ptr/copies to/of local intermeriary results
    all_local_max[id * BANKING_FACTOR] = DEFAULT_MAX_VALUE;
    all_local_indexes[id * BANKING_FACTOR] = NULL;

    // Initialize your local pointer of the tile lock
    tile_lock_ptr =
        &all_tiles_locks[tile_id * NUM_CORES_PER_TILE * BANKING_FACTOR];

    // Initialize your local vector of data somewhere in the tile
    do_lock(tile_lock_ptr);
    local_vector =
        (int32_t *)domain_malloc(tile_alloc, local_data_len * sizeof(int32_t));
    // If no more space in the tile then anywhere in L1
    if (!local_vector) {
      do_lock(&malloc_lock);
      local_vector = (int32_t *)simple_malloc(local_data_len * sizeof(int32_t));
      do_unlock(&malloc_lock);
      if (!local_vector)
        printf("ERROR\n");
    }
    do_unlock(tile_lock_ptr);

    // Fill your local vector of data
    uint32_t local_i = 0;
#pragma omp for
    for (uint32_t i = 0; i < data_len; ++i) {
      // Keep your offset for later
      if (local_i == 0)
        local_offset = i;

      local_vector[local_i++] = data[i];
    }

    mempool_stop_benchmark();
    mempool_start_benchmark();

    // Start argmax computation on local_vector
    for (uint32_t i = 0; i < local_data_len; ++i) {

      // We found a better maximum
      if (local_vector[i] > local_max) {
        // Set your local max to him
        local_max = local_vector[i];

        // Free your local index list
        do_lock(tile_lock_ptr);
        // Todo do not free all the time
        free_all(tile_alloc, local_indexes);
        // Start a new index list in the tile
        local_indexes = (index_t *)domain_malloc(tile_alloc, sizeof(index_t));
        // If no more space in the tile then anywhere in L1
        if (!local_indexes) {
          do_lock(&malloc_lock);
          local_indexes = (index_t *)simple_malloc(sizeof(index_t));
          if (!local_indexes)
            printf("ERROR\n");
          do_unlock(&malloc_lock);
        }
        do_unlock(tile_lock_ptr);

        // Save this new max's index
        local_indexes->idx = local_offset + i;
        local_indexes->next = NULL;

        local_indexes_len = 1;

        // We found the same maximum
      } else if (local_vector[i] == local_max) {

        // Allocate a new entry for its index
        index_t *new_index;
        do_lock(tile_lock_ptr);
        new_index = (index_t *)domain_malloc(tile_alloc, sizeof(index_t));
        if (!new_index) {
          do_lock(&malloc_lock);
          new_index = (index_t *)simple_malloc(sizeof(index_t));
          if (!new_index)
            printf("ERROR\n");
          do_unlock(&malloc_lock);
        }
        do_unlock(tile_lock_ptr);

        // Save this new max's index
        new_index->idx = local_offset + i;
        // Push this new max on the top of our result list
        new_index->next = local_indexes;
        local_indexes = new_index;

        local_indexes_len++;
      }
    }

    all_local_indexes[id * BANKING_FACTOR] = local_indexes;
    all_local_max[id * BANKING_FACTOR] = local_max;

    mempool_stop_benchmark();
    mempool_start_benchmark();

// Start log reduction
#if NUM_CORES_BENCH > 1
    int32_t winner_idx = mempool_log_reduction(2, id, num_cores);
    if (winner_idx >= 0) {
      *global_indexes = all_local_indexes[winner_idx];
      *global_max = all_local_max[winner_idx];
    }
#else
    *global_indexes = all_local_indexes[id * BANKING_FACTOR];
    *global_max = all_local_max[id * BANKING_FACTOR];
#endif
    mempool_stop_benchmark();
  }
}

////////////////////////////////////
/////// REDUCTION IMPLEMENTATION ///
////////////////////////////////////

int32_t mempool_log_reduction(uint32_t volatile step, uint32_t core_id,
                              uint32_t num_cores) {
  // TODO : Comments
  uint32_t idx_0, idx_1, step_idx = (step * (core_id / step)) * BANKING_FACTOR;
  uint32_t next_step, previous_step;

  previous_step = step >> 1;

  // Check if the collegue arrived before
  if ((step - previous_step) ==
      __atomic_fetch_add(&red_barrier[step_idx + previous_step - 1],
                         previous_step, __ATOMIC_RELAXED)) {
    // He did, so compare your values
    idx_0 = step_idx;
    idx_1 = idx_0 + previous_step * BANKING_FACTOR;

    // First one is greater
    if (all_local_max[idx_0] > all_local_max[idx_1]) {
      // Do nothing, all_local_max[step_idx] = all_local_max[idx_0]
    }
    // Second one is greater
    else if (all_local_max[idx_0] < all_local_max[idx_1]) {
      all_local_max[step_idx] = all_local_max[idx_1];
      all_local_indexes[step_idx] = all_local_indexes[idx_1];
    }
    // They are equal, point idx_1 indexes after idx_0 ones
    else {
      index_t *tmp = all_local_indexes[idx_0];
      while (tmp->next) {
        tmp = tmp->next;
      }
      tmp->next = all_local_indexes[idx_1];
    }

    next_step = step << 1;
    __atomic_store_n(&red_barrier[step_idx + previous_step - 1], 0,
                     __ATOMIC_RELAXED);

    if (step == num_cores) {
      // This is the winner
      __sync_synchronize(); // Full memory barrier
#if IS_MEMPOOL
      wake_up_all();
      mempool_wfi();
#endif
      return (int32_t)idx_0;
    } else {
      return mempool_log_reduction(next_step, core_id, num_cores);
    }

  } else {
#if IS_MEMPOOL
    mempool_wfi();
#endif
    return -1;
  }
}
