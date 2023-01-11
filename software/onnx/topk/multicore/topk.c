#include "include/topk.h"

/* STATIC DATA DECLARATION */
/* Shared global array containing the locks used to synchronize dynamic memory allocation. */
/* One lock per tile is needed, according to the memory allocation implementation.         */
uint32_t tile_locks[256 / CORES_PER_TILE] __attribute__((section(".l1")));

/* Shared global array that stores pointers to the local copies of the */
/* top-k heaps allocated on each core's local memory.                  */
heap_t *local_heaps[256] __attribute__((section(".l1")));

uint32_t volatile barriers[256] __attribute__((aligned(256 * 4), section(".l1")));

void log2_barrier(uint32_t core_id, volatile uint32_t step){
    
    uint32_t group_id = core_id / 64;
    uint32_t local_id = (core_id % 64) / step;

    uint32_t prev_step = step >> 1;
    uint32_t barrier_base = (prev_step - 1) * (256 / prev_step);
    uint32_t barrier_idx = barrier_base + (core_id / step);

    if(step == 2){
        
        if ((step - 1) == __atomic_fetch_add(&barriers[barrier_idx], 1, __ATOMIC_RELAXED)) {
            __atomic_store_n(&barriers[barrier_idx], 0, __ATOMIC_RELAXED);
            __sync_synchronize();
            wake_up((core_id / 2) * step + 1);
            wake_up((core_id / 2) * step);
        } 

        mempool_wfi();


    } else if (step < 64){

        uint32_t tiles_to_wakeup = step >> 2;
    
        if ((step - 1) == __atomic_fetch_add(&barriers[barrier_idx], 1, __ATOMIC_RELAXED)) {
            __atomic_store_n(&barriers[barrier_idx], 0, __ATOMIC_RELAXED);
            __sync_synchronize();
            wake_up_tile(group_id, ((1U << tiles_to_wakeup) - 1) << (tiles_to_wakeup * local_id));
        }

        mempool_wfi();

    } else {

        uint32_t groups_to_wakeup = step >> 6;
    
        if ((step - 1) == __atomic_fetch_add(&barriers[barrier_idx], 1, __ATOMIC_RELAXED)) {
            __atomic_store_n(&barriers[barrier_idx], 0, __ATOMIC_RELAXED);
            __sync_synchronize();
            wake_up_group(((1U << groups_to_wakeup) - 1) << (groups_to_wakeup * (core_id / step)));
        }

        mempool_wfi();

    }
}

void topk_largest(uint32_t *v, uint32_t n, heap_t *h) {

    uint32_t core_id = mempool_get_core_id();
    
    for(uint32_t i = 0; i < HEAP_MAX_SIZE; ++i){
        h->data[i] = 0;
    }

    for(uint32_t i = core_id * 4; i < n; i += 1024){
        for(uint32_t j = 0; j < 1024 / ACTIVE_CORES; ++j){
            if(v[i + j] > h->data[0]){
                h->data[0] = v[i + j];
                heapify_max(h, 0);
            }
        }
    }
}

void topk_smallest(uint32_t *v, uint32_t n, heap_t *h) {

    uint32_t core_id = mempool_get_core_id();

    for(uint32_t i = 0; i < HEAP_MAX_SIZE; ++i){
        h->data[i] = (uint32_t) 0xFFFFFFFF;
    }

    for(uint32_t i = core_id * 4; i < n; i += 1024){
        for(uint32_t j = 0; j < 1024 / ACTIVE_CORES; ++j){
            if(v[i + j] < h->data[0]){
                h->data[0] = v[i + j];
                heapify_min(h, 0);
            }
        }
    }
}

heap_t *topk(uint32_t *v, uint32_t n, uint32_t k, uint8_t largest) {
    /*
     * Top-K function: retrieve the largest/smallest elements in a given array of arbitrary size.
     * 
     * Parameters:
     * - v : pointer to an L1 memory region containing the input vector
     * - n : size of the array pointed by v
     * - k : number of elements to retrieve
     * - largest : whether to retrieve the largest or smallest elements
     * 
     * Return value:
     * - heap_t: a max-heap data structure containing the K largest elements of the array
    */


    uint32_t core_id, tile_id;

    core_id = mempool_get_core_id();
    tile_id = core_id / CORES_PER_TILE;

    alloc_t  *tile_alloc = get_alloc_tile(tile_id);
    uint32_t *tile_lock = &(tile_locks[tile_id]);

    /* Allocate memory in the local tile for the local copy of the largest K elements */
    lock_acquire(tile_lock);
    heap_t *h = (heap_t *) domain_malloc(tile_alloc, sizeof(heap_t));
    lock_release(tile_lock);

    /* Initialize heap */
    h->size = k;
    
    if(largest) {
        topk_largest(v, n, h);
    } else {
        topk_smallest(v, n, h);
    }

    return h;
}

void reduce_largest(uint32_t k, uint32_t core_id, volatile uint32_t step) {
    
    uint32_t prev_step = step >> 1;
    uint32_t next_step = step << 1;

    heap_t *h = local_heaps[core_id];

    if((core_id % step) == 0){
        heap_t *h_neighbour = local_heaps[core_id + prev_step];
        for(int32_t j = (int32_t)k - 1; j >= 0; j--){
            if(h_neighbour->data[j] > h->data[0]){
                h->data[0] = h_neighbour->data[j];
                heapify_max(h, 0);
            }
        }
    }

    if(next_step <= 256){
        log2_barrier(core_id, next_step);
        reduce_largest(k, core_id, next_step);
    }
}

void reduce_smallest(uint32_t k, uint32_t core_id, volatile uint32_t step) {
    
    uint32_t prev_step = step >> 1;
    uint32_t next_step = step << 1;

    heap_t *h = local_heaps[core_id];

    if((core_id % step) == 0){
        heap_t *h_neighbour = local_heaps[core_id + prev_step];
        for(int32_t j = (int32_t)k - 1; j >= 0; j--){
            if(h_neighbour->data[j] < h->data[0]){
                h->data[0] = h_neighbour->data[j];
                heapify_min(h, 0);
            }
        }
    }

    if(next_step <= 256){
        log2_barrier(core_id, next_step);
        reduce_smallest(k, core_id, next_step);
    }
}

void reduce(uint32_t k, uint32_t core_id, volatile uint32_t step, uint8_t largest) {
    /*
     * Logarithmic reduction
     * Performs reduction over the 256 cores of Mempool recursively. The runtime of this function
     * grows linearly with K, and logarithmically with the number of cores.
     * 
     * Parameters:
     * - k       : number of elements of each heap to reduce. This value greately influences 
     *             the performance of this function.
     * - core_id : core ID of the core executing this function (avoids multiple calls to 
     *             get_core_id()).
     * - step    : represents the current progress in the recursive reduction algorithm.
     *             Each iteration, step represents the number of cores over which the 
     *             result is being reduced. For example, with 256 cores, the first iteration
     *             should have step = 2. The step increases as powers of 2 since it is a log2 
     *             reduction.
     * - largest : whether to retrieve the largest or smallest elements
     * 
     * Return value:
     * - void
    */

    if(largest) {
        reduce_largest(k, core_id, step);
    } else {
        reduce_smallest(k, core_id, step);
    }
}
