#ifndef DPHCP_TOPK_H
#define DPHCP_TOPK_H

#include <stdint.h>
#include "alloc.h"
#include "runtime.h"
#include "synchronization.h"
#include "config.h"
#include "heap.h"
#include "lock.h"

extern uint32_t tile_locks[];
extern heap_t  *local_heaps[];
extern uint32_t volatile barriers[]; 


heap_t *topk(uint32_t *v, uint32_t n, uint32_t k, uint8_t largest);
void reduce(uint32_t k, uint32_t core_id, volatile uint32_t step, uint8_t largest);

#endif