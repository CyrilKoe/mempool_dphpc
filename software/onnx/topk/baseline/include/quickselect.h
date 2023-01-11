#ifndef QUICKSELECT_H
#define QUICKSELECT_H

#include <stdint.h>

uint32_t partition(uint32_t *, uint32_t, uint32_t, uint32_t);
void quickselect(uint32_t *, uint32_t, uint32_t, uint32_t);

#endif // QUICKSELECT_H