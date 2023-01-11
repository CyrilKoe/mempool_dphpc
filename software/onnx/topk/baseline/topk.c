#include "include/topk.h"

/* STATIC DATA DECLARATION */
/* Shared global array containing the locks used to synchronize dynamic memory allocation. */
/* One lock per tile is needed, according to the memory allocation implementation.         */
uint32_t tile_locks[256 / CORES_PER_TILE] __attribute__((section(".l1")));

heap_t *topk_heap(uint32_t *v, uint32_t n, uint32_t k) {
    /*
     * Top-K function: retrieve the largest elements in a given array of arbitrary size.
     * 
     * Parameters:
     * - v : pointer to an L1 memory region containing the input vector
     * - n : size of the array pointed by v
     * - k : number of elements to retrieve
     * 
     * Return value:
     * - heap_t: a max-heap data structure containing the K largest elements of the array
    */

    alloc_t  *tile_alloc = get_alloc_tile(0);
    uint32_t *tile_lock = &(tile_locks[0]);

    /* Allocate memory in the local tile for the local copy of the largest K elements */
    lock_acquire(tile_lock);
    heap_t *h = (heap_t *) domain_malloc(tile_alloc, sizeof(heap_t));
    lock_release(tile_lock);
	
    /* Initialize heap */
    h->size = k;
    for(uint32_t i = 0; i < HEAP_MAX_SIZE; ++i){
        h->data[i] = 0;
    }

    for(uint32_t i = 0; i < n; ++i){
        if(v[i] > h->data[0]){
            h->data[0] = v[i];
            heapify(h, 0);
        }
    }

    return h;
}

void topk_quickselect(uint32_t *v, uint32_t n, uint32_t k) {

    // run quickselect
    quickselect(v, 0, n, k);

    return;
}
