#include <stdlib.h>
#include <stdint.h>

static inline void swap(uint32_t *i, uint32_t*j){
    uint32_t tmp = *i;
    *i = *j;
    *j = tmp;
}

uint32_t partition(uint32_t *v, uint32_t start, uint32_t stop, uint32_t pivot_index) {

    uint32_t pivot_value = v[pivot_index];

    swap(&v[pivot_index], &v[stop - 1]);

    uint32_t store_index = start;

    for(uint32_t i = start; i < stop; ++i){

        if(v[i] > pivot_value){
            swap(&v[store_index], &v[i]);
            store_index += 1;
        }
    }

    swap(&v[store_index], &v[stop - 1]);

    return store_index;
}

void quickselect(uint32_t *v, uint32_t start, uint32_t stop, uint32_t k) {

    if(start >= stop - 1){
        return;
    }

    uint32_t pivot_index = start + (((uint32_t)rand()) % (stop - start)); // stop - 1; // start + (uint32_t)(rand() % (int)(stop - start));

    pivot_index = partition(v, start, stop, pivot_index);

    if(k == pivot_index){
        return;
    }

    if(k < pivot_index){
        quickselect(v, start, pivot_index, k);
    } else {
        quickselect(v, pivot_index + 1, stop, k);
    }
}