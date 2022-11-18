
#include <inttypes.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "data.h"

#include "alloc.h"
#include "encoding.h"
#include "libgomp.h"
#include "printf.h"
#include "runtime.h"
#include "synchronization.h"
#define NUM_BANKS (NUM_CORES *BANKING_FACTOR)
#define DEFAULT_MAX_VALUE -(int32_t)(1 << 16);

uint32_t *print_lock;
uint32_t print_lock_l2 __attribute__((section(".l2")));
uint32_t print_lock_l1 __attribute__((section(".l1")));

void lock_tile(uint32_t *lock) {
  uint32_t islocked;
  islocked = __atomic_fetch_or(lock, 1, __ATOMIC_SEQ_CST);
  while (islocked) {
    mempool_wait(NUM_CORES_PER_TILE);
    islocked = __atomic_fetch_or(lock, 1, __ATOMIC_SEQ_CST);
  }
}

void unlock_tile(uint32_t *lock) {
  __atomic_fetch_and(lock, 0, __ATOMIC_SEQ_CST);
}


int main() {

  uint32_t core_id = mempool_get_core_id();
  uint32_t num_cores = mempool_get_core_count();
  mempool_barrier_init(core_id);
  mempool_init(core_id, num_cores);
  if (core_id == 0) {
    print_lock = simple_malloc(sizeof(uint32_t));
    *print_lock = 0;

#pragma omp parallel num_threads(NUM_CORES)
    {
      uint32_t id = omp_get_thread_num();
      //uint32_t tile_id = id / NUM_CORES_PER_TILE;

      if (id < 32) {
      lock_tile(print_lock);
      printf("VERY LONG STRING %u\n", id);
      unlock_tile(print_lock);
      }

      #pragma omp barrier

      if (id < 32) {
      lock_tile(&print_lock_l2);
      printf("VERY LONG STRING %u\n", id);
      unlock_tile(&print_lock_l2);
      }

      #pragma omp barrier

      if (id < 32) {
      lock_tile(&print_lock_l1);
      printf("VERY LONG STRING %u\n", id);
      unlock_tile(&print_lock_l1);
      }

    }


  } else {
    while (1) {
      mempool_wfi();
      run_task(core_id);
    }
  }
  
}