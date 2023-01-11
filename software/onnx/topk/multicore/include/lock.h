#ifndef DPHCP_LOCK_H
#define DPHCP_LOCK_H

#include "config.h"

extern void lock_acquire(uint32_t *lock);
extern void lock_release(uint32_t *lock);

#endif