#ifndef HEAP_H
#define HEAP_H

#define HEAP_MAX_SIZE 64

typedef struct heap {
	uint32_t data[HEAP_MAX_SIZE];
	uint32_t size;
} heap_t;

// extern inline void init_heap(heap_t *h, uint32_t size);
void heapify(heap_t *, uint32_t);

#endif // HEAP_H