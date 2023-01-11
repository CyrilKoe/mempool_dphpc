/* STANDARD LIBRARIES */
#include <stdint.h>

/* MEMPOOL LIBRARIES */
#include "dma.h"
#include "alloc.h"
#include "printf.h"
#include "runtime.h"
#include "encoding.h"
#include "synchronization.h"
#include "mempool_dma_frontend.h"

/* LOCAL LIBRARIES */
#include "include/config.h"
#include "include/heap.h"
#include "include/lock.h"
#include "include/data.h"
#include "include/topk.h"

#define DEBUG 1
#define BENCH_TOPK 0
#define USE_HEAP 1

/*
  * L1 vector where to load the input data. The vector is placed at the beginning of the 
  * interleaved region of L1 memory, and aligned in a way such that the first elements 
  * are placed in bank #0 of the first tile, and so on. This guarantees that each core 
  * can perform only local memory accesses by using an appropriate memory access pattern.
*/
uint32_t v[N] __attribute__((section(".l1_prio"), aligned(1024 * 4)));

int main(){

    uint32_t core_id, num_cores, tile_id;

    core_id   = mempool_get_core_id();
    num_cores = mempool_get_core_count();
    tile_id   = core_id / CORES_PER_TILE;

    mempool_init(core_id, num_cores);
    mempool_barrier_init(core_id);

    mempool_barrier(num_cores);

    /* One core per tile initializes the tile lock */
    if(core_id % CORES_PER_TILE == 0){
        tile_locks[tile_id] = 0;
    }

    #if USE_HEAP
    heap_t *local_heap = (heap_t *) 0;
    #endif

    mempool_barrier(num_cores);

    #if DEBUG
    if(core_id == 0) printf("Copying data into L1\n");
    #endif

    /* Load data into L1 */
    if(core_id == 0){
        dma_memcpy_blocking(v, input, N * 4);
    }

    mempool_barrier(num_cores);

    #if DEBUG
    if(core_id == 0) printf("Starting\n");
    #endif

    #if BENCH_TOPK
    mempool_barrier(num_cores);
    mempool_start_benchmark();
    #endif

    /* 
     * Run Top-K on CORE 0:
    */
    if(core_id == 0){
        #if USE_HEAP
        local_heap = topk_heap(v, N, K);
        #else
        topk_quickselect(v, N, K);
        #endif
    }

    mempool_barrier(num_cores);

    #if BENCH_TOPK
    mempool_stop_benchmark();
    mempool_barrier();
    #endif

    #if DEBUG
    /* show results */
    
    #if USE_HEAP
    uint32_t *output = &(local_heap->data[0]);
    #else
    uint32_t *output = &(v[0]);
    #endif

    if(core_id == 0){
       	printf("TopK elements: [ ");
       	for(uint32_t i = 0; i < K - 1; ++i){
           	printf("%u, ", output[i]);
       	}
       	printf("%u ]\n", output[K-1]);

        uint32_t errors = 0;
        for(uint32_t i = 0; i < K; ++i) {
            uint32_t el = output[i];
            uint32_t found = 0;
            /* WARNING:
             * To make this check work, there should be an array of K elements 
             * somewhere containing the expected values. This vector can be 
             * generated from the golden model.
            */
            for(uint32_t j = 0; j < K; j++){
                if(expected_output[j] == el){
                    found = 1;
                    break;
                }
            }
            if(found == 0){
                errors += 1;
            }
        }
        printf("%u errors\n", errors);
    }

	mempool_barrier(num_cores);
    #endif

	return 0;
}