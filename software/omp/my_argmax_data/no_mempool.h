#ifndef NO_MEMPOOL_H_
#define NO_MEMPOOL_H_

#define NUM_CORES_PER_TILE 4
#define NUM_CORES 256
#define BANKING_FACTOR 4

typedef struct {
  char nothing;
} alloc_t;

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

void mempool_stop_benchmark() {;}
void mempool_start_benchmark() {;}



#endif