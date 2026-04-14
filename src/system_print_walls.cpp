/******************************************************
This code has been developed by:
Adolfo Vazquez-Quesada (1) and Jose Manuel Moreno Valderrama (2)
(1) Department of Fundamental Physics at UNED, Madrid, Spain
(2) Remedy Entertainment
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/

#include <iostream>
#include "cuda_runtime.h"
#include "class_system.h"

// function to print walls info
void class_system::print_walls(real* k_fx_wall,
			       real* k_fy_wall,
			       real* k_fz_wall,
			       int step) {

  char filename[50];
  sprintf(filename, "walls-%d.dat", step);
  FILE* file = fopen(filename, "w");
  if (!file) {
    printf("System print walls error: Error opening the file %s\n", filename);
    return;
  }
  
  cudaMemcpy(this->fx_wall, k_fx_wall, Nwalls * sizeof(real), cudaMemcpyDeviceToHost);
  cudaMemcpy(this->fy_wall, k_fy_wall, Nwalls * sizeof(real), cudaMemcpyDeviceToHost);
  if (dim == 3)
    cudaMemcpy(this->fz_wall, k_fz_wall, Nwalls * sizeof(real), cudaMemcpyDeviceToHost);

  if (dim == 2)
    for (int i = 0; i < Nwalls; ++i)
      fprintf(file, "%d " REAL_FMT " " REAL_FMT "\n",
	      i,
	      this->fx_wall[i],
	      this->fy_wall[i]);
  else //--- dim == 3  ---
    for (int i = 0; i < Nwalls; ++i)
      fprintf(file, "%d " REAL_FMT " " REAL_FMT " " REAL_FMT "\n",
	      i,
	      this->fx_wall[i],
	      this->fy_wall[i],
	      this->fz_wall[i]);    
  fclose(file);

  printf("Walls file written. Time step %d\n", step);
}
