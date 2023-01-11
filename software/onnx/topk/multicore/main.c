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
#define BENCH_REDUCE 0
#define LARGEST 1
#define SORT 0

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

    mempool_init(core_id);
    mempool_barrier_init(core_id);

    mempool_barrier(num_cores);

    /* Initialize barriers (for log-barrier) */
    if(core_id == 0){
        for(uint32_t i = 0; i < 256; ++i){
            barriers[i] = 0;
        }
    }

    /* One core per tile initializes the tile lock */
    if(core_id % CORES_PER_TILE == 0){
        tile_locks[tile_id] = 0;
    }

    local_heaps[core_id] = (heap_t *) 0;

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
     * Run Top-K on active cores:
     * Active cores are chosen to be non-contiguous ones (if ACTIVE_CORE_RANGE < 256), 
     * so that the amount of local memory for each core is maximized, and bank conflicts
     * are avoided.
    */
    if(core_id % ACTIVE_CORE_RANGE == 0){
    	local_heaps[core_id] = topk(v, N, K, LARGEST);
    }

    mempool_barrier(num_cores);

    #if BENCH_TOPK
    mempool_stop_benchmark();
    #endif

    #if DEBUG
    if(core_id == 0) printf("Reduction\n");
    #endif

    #if BENCH_REDUCE
    mempool_barrier(num_cores);
    mempool_start_benchmark();
    #endif

	/* Perform log-reduction */
    #if (ACTIVE_CORES > 1)
	reduce(K, core_id, 2*256/ACTIVE_CORES, LARGEST);
    #endif

	mempool_barrier(num_cores);

    #if BENCH_REDUCE
    mempool_stop_benchmark();
    #endif

    #if SORT
    if(core_id == 0) heapsort(local_heaps[core_id]);
    mempool_barrier(num_cores);
    #endif

    #if DEBUG
    /* show results */
    if(core_id == 0){
       	printf("TopK elements: [ ");
       	for(uint32_t i = 0; i < K - 1; ++i){
           	printf("%u, ", local_heaps[0]->data[i]);
       	}
       	printf("%u ]\n", local_heaps[0]->data[K-1]);

        uint32_t errors = 0;
        for(uint32_t i = 0; i < K; ++i) {
            uint32_t el = local_heaps[0]->data[i];
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