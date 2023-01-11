#include <stdint.h>
#include "runtime.h"
#include "include/config.h"
#include "include/lock.h"

///////////////////////////////////////////////////////////////////
///////////////////////// MEMPOOL LOCKS ///////////////////////////
///////////////////////////////////////////////////////////////////

/* 
 * Simple implementation of memory locks. These functions require a 
 * pointer to an integer used to store the lock value. Thet can 
 * therefore be shared among locks used for different purposes (e.g.
 * printing, allocating memory, barriers, ...).
*/

inline void lock_acquire(uint32_t *lock) {
  uint32_t islocked;
  islocked = __atomic_fetch_or(lock, 1, __ATOMIC_SEQ_CST);
  while (islocked) {
    mempool_wait(CORES_PER_TILE * 5);
    islocked = __atomic_fetch_or(lock, 1, __ATOMIC_SEQ_CST);
  }
}

inline void lock_release(uint32_t *lock) {
  __atomic_fetch_and(lock, 0, __ATOMIC_SEQ_CST);
}