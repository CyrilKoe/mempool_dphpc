#ifndef DPHPC_DATA_H
#define DPHPC_DATA_H

#include <stdint.h>

#define N ${len(values)}
<%
    def data_type(precision, signed):
        if precision == 8:
            return 'int8_t' if signed else 'uint8_t'
        elif precision == 16:
            return 'int16_t' if signed else 'uint16_t'
        elif precision == 32:
            return 'int32_t' if signed else 'uint32_t'
        else:
            raise Exception("wrong precision")
%>
${data_type(precision, signed)} v[N] __attribute__((section(".l1"), aligned(1024 * 4))) = {
% for val in values:
    ${val},
% endfor
};

#endif