#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include "include/data.h"

#define HEAP_MAX_SIZE 1000

typedef struct heap {
	uint32_t data[HEAP_MAX_SIZE];
	uint32_t size;
} heap_t;

static inline void swap(uint32_t *i, uint32_t*j){
    uint32_t tmp = *i;
    *i = *j;
    *j = tmp;
}

static inline int parent(int i){
    return (i - 1) / 2;
}

static inline int leftchild(int i){
    return i * 2 + 1;
}

static inline int rightchild(int i){
    return i * 2 + 2;
}

void heapify(heap_t *heap, int i){

    int parent_idx = i;
    int lchild_idx = leftchild(parent_idx);
    int rchild_idx = rightchild(parent_idx);

    uint32_t heap_size = heap->size;

    uint32_t parent = heap->data[parent_idx];

    int smallest = parent_idx;

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

uint32_t getmin(heap_t *heap){
	
	return heap->data[0];
}

void printheap(heap_t *heap){
    printf("[ ");
    for(uint32_t i = 0; i < heap->size - 1; ++i){
        printf("%d, ", heap->data[i]);
    }
    printf("%d ]\n", heap->data[heap->size - 1]);
}

void insert(heap_t *heap, uint32_t v){

	if(v <= getmin(heap)){
		return;
	}

	heap->data[0] = v;

	heapify(heap, 0);
}

uint32_t *topk(uint32_t *v, uint32_t n, uint32_t k){

	heap_t heap;
	
	memset((void *) &(heap.data), 0, HEAP_MAX_SIZE * sizeof(uint32_t));
	heap.size = k;

	clock_t start = clock();

	for(uint32_t i = 0; i < n; ++i){
		insert(&heap, v[i]);
	}

	clock_t end = clock();

	double cycles = (double)(end - start) / CLOCKS_PER_SEC; 

	printf("Execution time: %.2f\n", cycles);

	uint32_t *vout = malloc(k * sizeof(uint32_t));

	memcpy(vout, &heap.data[0], k * sizeof(uint32_t));

	return vout;
}

int main(){

	uint32_t *top = topk(v, N, K);

    printf("Top %d elements found: [", K);
    for(uint32_t i = 0; i < K - 1; ++i){
        printf("%u, ", top[i]);
    }
    printf("%u ]\n", top[K-1]);

	return 0;
}