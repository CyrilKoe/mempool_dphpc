#ifndef HEAP_H
#define HEAP_H

#define HEAP_MAX_SIZE 64

typedef struct heap {
	uint32_t data[HEAP_MAX_SIZE];
	uint32_t size;
} heap_t;

// extern inline void init_heap(heap_t *h, uint32_t size);
void heapify_max(heap_t *, uint32_t);
void heapify_min(heap_t *, uint32_t);
void heapsort(heap_t *heap);

#endif // HEAP_H