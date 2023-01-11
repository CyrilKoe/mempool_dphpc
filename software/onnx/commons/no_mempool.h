#ifndef NO_MEMPOOL_H_
#define NO_MEMPOOL_H_

#define NUM_CORES_PER_TILE 4
#define NUM_CORES 256
#define BANKING_FACTOR 4


#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <unistd.h>


// Linked list to measure regions's cpu time
typedef struct measure {
    clock_t start;
    clock_t end;
    struct measure *next;
} measure_t;

measure_t *measure_list = NULL;

// Fake alloc stuct to fit with mempool library
typedef struct {
  char nothing;
} alloc_t;

typedef clock_t mempool_timer_t;

static inline mempool_timer_t mempool_get_timer() { return clock(); }
static inline void mempool_wait(uint32_t cycles) { usleep(cycles); }


void domain_free(alloc_t *alloc, void *const ptr) {
    free(ptr);
}

void *domain_malloc(alloc_t *alloc, const uint32_t size) {
    return malloc(size);
}

void *simple_malloc(const uint32_t size) {
    return malloc(size);
}

void *simple_free(void *const ptr) {
    free(ptr);
}

alloc_t *get_alloc_tile(const uint32_t tile_id) { return NULL; }

void mempool_start_benchmark() {
    measure_t *tmp = measure_list;
    measure_t *new_elem;

    new_elem = (measure_t*) malloc(sizeof(measure_t));
    new_elem->next = NULL;

    if(measure_list == NULL) {
        measure_list = new_elem;
        new_elem->start = clock();
        return;
    }

    while(tmp->next)
        tmp = tmp->next;
    
    tmp->next = new_elem;
    new_elem->start = clock();
    return;
}

void mempool_stop_benchmark() {
    measure_t *tmp = measure_list;
    long end = clock();
    while(tmp->next)
        tmp = tmp->next;
    tmp->end = end;
    return;
}

void print_benchmark() {
    measure_t *tmp = measure_list;
    unsigned int i = 0;
    while(tmp) {
        printf("Section %u : %lu cycles (%lu -> %lu)\n", i++, tmp->end - tmp->start, tmp->start, tmp->end);
        tmp = tmp->next;
    }
    printf("Cycles per sec %lu\n", CLOCKS_PER_SEC);
    return;
}

#endif