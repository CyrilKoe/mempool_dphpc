#include <stdint.h>
#include "include/heap.h"

// inline void init_heap(heap_t *h, uint32_t size){
//     h->size = size;
//     for(uint32_t i = 0; i < HEAP_MAX_SIZE; ++i){
//         h->data[i] = 0;
//     }
// }

static inline void swap(uint32_t *i, uint32_t*j){
    uint32_t tmp = *i;
    *i = *j;
    *j = tmp;
}

static inline uint32_t leftchild(uint32_t i){
    return i * 2 + 1;
}

static inline uint32_t rightchild(uint32_t i){
    return i * 2 + 2;
}

void heapify(heap_t *heap, uint32_t i){
    /*
     * Heapify function: takes a heap and the index of a node of it, 
     * and moves it to the correct position in the heap.
     * 
     * Parameters:
     * - heap : pointer to the data structure to heapify.
     * - i    : index of the element to heapify (e.g. 0 for the root)
     * 
     * Return value:
     * - void
    */

    uint32_t parent_idx = i;
    uint32_t lchild_idx = leftchild(parent_idx);
    uint32_t rchild_idx = rightchild(parent_idx);

    uint32_t heap_size = heap->size;

    uint32_t parent = heap->data[parent_idx];

    uint32_t smallest = parent_idx;

    if(lchild_idx < heap_size){
    	uint32_t lchild = heap->data[lchild_idx];
    	if(parent > lchild){
    		smallest = lchild_idx;
    	}
    }
    if(rchild_idx < heap_size){
    	uint32_t rchild = heap->data[rchild_idx];
    	if(heap->data[smallest] > rchild){
    		smallest = rchild_idx;
    	}
    }
    if(smallest != i){
    	swap(&heap->data[i], &heap->data[smallest]);
    	heapify(heap, smallest);
    }
}