/******************************************************
This code has been developed by:
Adolfo Vazquez-Quesada (1) and Jose Manuel Moreno Valderrama (2)
(1) Department of Fundamental Physics at UNED, Madrid, Spain
(2) Remedy Entertainment
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/

#include "kernel_functions.h"
#include "config.h"

//Function to set total kinetic energy and total momentum to zero
__global__ void kernel_macro_vars_to_zero(real* __restrict__ kin_energy) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= N) return;

  *kin_energy = 0.0;
}
