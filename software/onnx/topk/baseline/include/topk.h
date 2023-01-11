#ifndef DPHCP_TOPK_H
#define DPHCP_TOPK_H

#include <stdint.h>
#include "alloc.h"
#include "runtime.h"
#include "synchronization.h"
#include "config.h"
#include "heap.h"
#include "quickselect.h"
#include "lock.h"

extern uint32_t tile_locks[];

heap_t *topk_heap(uint32_t *v, uint32_t n, uint32_t k);
void topk_quickselect(uint32_t *v, uint32_t n, uint32_t k);

#endif