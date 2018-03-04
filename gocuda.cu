/* Simple CUDA Example -- Williams */
#include <iostream>
#include <math.h>
#include <stdio.h>

// __global__ means this function is available on CPU and GPU


// This version does NOT print any data out for debugging
__global__
void scale(unsigned int n, float *x, float *y)
{
  unsigned int i, base=blockIdx.x*blockDim.x+threadIdx.x, incr=blockDim.x*gridDim.x;
  for (i=base;i<n;i+=incr)  // note that i>=n is discarded
    x[i]=x[i]*y[i];

}


// This version DOES print data out for debugging
__global__
void scaleprint(unsigned int n, float *x, float *y)
{
  unsigned int i, base=blockIdx.x*blockDim.x+threadIdx.x, incr=blockDim.x*gridDim.x;
  printf("t=%x: base=%d incr=%d n=%d block=%d\n",threadIdx.x, base, incr, n,blockIdx.x);	
  for (i=base;i<n;i+=incr)  // note that i>=n is discarded
    x[i]=x[i]*y[i];

}




// Configurations
// N is the data set size
int N=16;  // do not make this unsigned because we use -N => 1<<|N|
// Default block size
unsigned int blksize=512;
// Number of blocks
unsigned int nblock=0;

// Note, if N=blksize*nblock then you will get a grid of 1 and the kernels will each process one data item
// However, if N>blksize*nblock then each kernel will run its loop and process more than one element


// options
int nonorm=0;  // option - 1 means no CPU run (implies notest)
int nocuda=0;  // option - 1 means no GPU run (implies notest)
int notest=0;  // option - 1 means don't compare results
int print=0;   // option - 1 means use printing kernel
int enddebug=0;  // option - 1 means print end debug

int process_options(int,char *[]);


int main(int argc, char *argv[])
{
  float *x;  // input vector
  float *y;  // output vector
  float *testvec;  // vector for testing (CPU computed)

  // read command line
  int rv=process_options(argc, argv);
  if (rv) return rv;
  

// We need shared memory that the CPU and GPU can both access
  std::cout<<"N="<<N<<std::endl;  // allocate shared memory
  cudaMallocManaged(&x,N*sizeof(float));
  cudaMallocManaged(&y,N*sizeof(float));
  if (nonorm==0) testvec=(float *)malloc(N*sizeof(float));
  
// Generate input vectors
  for (unsigned int i=0;i<N;i++)
    {
      x[i]=1.0f;
      y[i]=(i%10)/10.0f;
// If we are supposed to do a normal run, do it now
      if (nonorm==0)
         testvec[i]=x[i]*y[i];  // compute right answer old fashioned way
    }
  if (nonorm==1) std::cout<<"Normal skipped"<<std::endl;
// Do cuda run unless disabled
  if (nocuda==0)
    {
    int numblk=(N+blksize-1)/blksize; // Number of whole blocks required to contain N
    if (nblock) numblk=nblock;  // override if provided
    std::cout<<"Start kernel "<<blksize<<" "<<numblk<<std::endl;
// This is the line that kicks off the kernel
     if (print)
          scaleprint<<<numblk,blksize>>>((unsigned int)N,x,y);
     else
          scale<<<numblk,blksize>>>((unsigned int)N,x,y);
// Wait for processing to complete
     cudaDeviceSynchronize();
    } else std::cout<<"Cuda skipped"<<std::endl;
// Compare results if asked to do so
  if (notest==0)
    {
      for (unsigned int i=0;i<N;i++)
       {
         if (x[i]!=testvec[i])
	  {
	    std::cout<<"Error at "<<i<<std::endl;
	  }
       }
     if (testvec) free(testvec);
     std::cout<<"Test Complete"<<std::endl;     
  }
// Dump some results
  if (enddebug==1)
  {
    std::cout<<"First 16 elements:"<<std::endl;
    for (unsigned int i=0;i<16;i++) std::cout<<(nocuda==0?x[i]:testvec[i])<<" ";
    std::cout<<std::endl;
  }
// release memory
  cudaFree(x);
  cudaFree(y);
  return 0;
}

      
      
// Process options
int process_options(int argc, char *argv[])
{
// Process options
  while (argc-->1)
    {
    argv++;
      if (**argv=='?')
	{
	  std::cout<<"Usage: gocuda [g|c] [p] [d] [bs=block_size] [nb=num_blocks] [number_of_samples]"<<std::endl;
	  std::cout<<"   g=GPU only; c=CPU only; d=end debugging dump; p=print inside kernel (for gpu mode)"<<std::endl;
	  std::cout<<"   if number of samples is negative, use 2**|number_of_samples|"<<std::endl;
	  std::cout<<"For example, gocuda -4 produces 16 samples (2**4)"<<std::endl;
	  return 1;
	}
      if (**argv=='d')  // print debug output at end
      {
        enddebug=1;
	continue;
      }
      if (**argv=='p')  // using printing kernel
      {
        print=1;
	continue;
      }
      if (**argv=='c') // no CUDA
        {
	  notest=nocuda=1;
	  continue;
        }
      if (**argv=='g')  // no CPU
	{
	  notest=nonorm=1;
	  continue;
	}
      if (argv[0][0]=='b' && argv[0][1]=='s' && argv[0][2]=='=' )  // block size
	{
	  blksize=atoi(*argv+3);
	  if (blksize<=0)
	    {
	      std::cout<<"Error: Blocksize must be positive!"<<std::endl;
	      return 3;
	    }
	  continue;
	}
      if (argv[0][0]=='n' && argv[0][1]=='b' && argv[0][2]=='=' )  // # of blocks
      {
         nblock=atoi(*argv+3);
	 if (nblock==0) std::cout<<"Warning: Number of blocks zero; will be auto calculated"<<std::endl;
	 continue;
      }
      N=atoi(*argv);
      if (N<0&&N>-31) N=1<<-N;   // if N is negative, take N as 1<<|N|
      if (N<=0)
      {
	std::cout<<"Error: N must be non-zero!"<<std::endl;
	return 2;
      }
    }
  return 0;
}


