#!/usr/bin/env python
import time
import numpy as np

from pycuda import gpuarray
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

import math

class MatrixMultiply:
    def __init__(self):
        """
        Attributes for instance of deviceAdd module
        Includes kernel code and input variables.
        """
        # host variables
        
        # tile parameters
        #self.tile_x = 2# tile size x
        #self.tile_y = # when changing this, remember to change it in kernel as well!
        
        matrix_mul_naive_kernel_code = """
        __global__ void Matrix_multiply_naive(float *a, float *b, float *c,
        const unsigned int M, const unsigned int N)
        {
            int col = blockDim.x * blockIdx.x + threadIdx.x;
            int row = blockDim.y * blockIdx.y + threadIdx.y;
            
            if (col < M && row < M)
            {
            float Pvalue = 0;
            int index = row*M + col;
            for(int k=0; k<N;k++){
            Pvalue += a[k + row*N] * b[k*M + col];
            }
            c[index] = Pvalue;
            }
        }
        """

        matrix_mul_optimized1_kernel_code = """
__global__ void Matrix_multiply_optimized1(float *a, float *b, float *c,
        const unsigned int M, const unsigned int N)
        {
            //'B' tiles are shared. 'A' tiles are in global memory 
            
            int col = blockDim.x * blockIdx.x + threadIdx.x;
            int row = blockDim.y * blockIdx.y + threadIdx.y;
            
            int ty = threadIdx.y;
            int tx = threadIdx.x;
            
            
            
             const int TILE_WIDTH = 2;
            __shared__ float b_share[TILE_WIDTH*TILE_WIDTH];
            
            float Pvalue = 0;
            for (int ph=0;ph < N/((float)TILE_WIDTH);ph++){
            if (col < M && (ph*TILE_WIDTH + ty)< N)
            {
            b_share[ty*TILE_WIDTH + tx] = b[col + M*(ph*TILE_WIDTH + ty)];
            }
            else
            {
            b_share[ty*TILE_WIDTH + tx] = 0;
            }
            __syncthreads();
            
            
            
            for (int k=0;k<TILE_WIDTH;k++)
            
            {
            
            Pvalue += a[row*N + ph*TILE_WIDTH + k] * b_share[k*TILE_WIDTH + tx];
            
            }
            __syncthreads();
            
            }
            if (row < M && col < M)
            {
            c[row*M+col] = Pvalue;
            }
            
        }
        """

        matrix_mul_optimized2_kernel_code = """
            __global__ void Matrix_multiply_optimized2(float *a, float *b, float *c,
        const unsigned int M, const unsigned int N)
        {
            const int TILE_WIDTH = 2;
            
            int row = blockIdx.y * blockDim.y  + threadIdx.y;
            int col= blockIdx.x  * blockDim.x  + threadIdx.x;
            
            int tx = threadIdx.x;
            int ty = threadIdx.y;
        
            
            __shared__ float a_shared[TILE_WIDTH*TILE_WIDTH];
            __shared__ float b_shared[TILE_WIDTH*TILE_WIDTH];
            
            float Pvalue = 0;
            for(int ph = 0; ph < N/((float)TILE_WIDTH);ph++)
            {
           if (row < M && (ph*TILE_WIDTH) < N){
            a_shared[ty*TILE_WIDTH+tx] = a[row*N + ph*TILE_WIDTH +tx];
            }
           else
           {
            a_shared[ty*TILE_WIDTH+tx] = 0;
           }
           if (col< M && (ph*TILE_WIDTH + ty)<N)
           {
            b_shared[ty*TILE_WIDTH + tx] = b[col + ph*TILE_WIDTH*M + ty*M];
            }
            else
            {
            b_shared[ty*TILE_WIDTH + tx] = 0;
            }
        
            __syncthreads();  
            
            for(int k=0;k<TILE_WIDTH;k++)
            {
            
            Pvalue += a_shared[ty*TILE_WIDTH+k] * b_shared[k*TILE_WIDTH + tx];
            }
            __syncthreads();
            }
            if (row < M && col < M)
            c[row*M + col] = Pvalue;
            
        }
        """
        # Build kernel codes (x3)
        self.mod = SourceModule(matrix_mul_naive_kernel_code)
        self.mod1 = SourceModule(matrix_mul_optimized1_kernel_code)
        self.mod2 = SourceModule(matrix_mul_optimized2_kernel_code)
        
    def matrix_mul_naive(self, a_cpu, b_cpu):
        """
        Function to perform matrix multipication. Should return the result
        and execution time.
        """
        
        func = self.mod.get_function("Matrix_multiply_naive")
        bdim_x = int(math.ceil(a_cpu.shape[0]/2))
        bdim_y = int(math.ceil(a_cpu.shape[1]/2))
        c = np.zeros((a_cpu.shape[0],a_cpu.shape[0]),dtype=np.float32)
        if bdim_x >= bdim_y:
            bdim = bdim_x
        else:
            bdim = bdim_y
        block_dim = (2,2,1)
        grid_dim = (bdim,bdim,1)
        e_start = cuda.Event()
        e_end = cuda.Event()
        e_start.record()
        pycuda.driver.Context.synchronize()
        a_gpu = cuda.mem_alloc(a_cpu.size * a_cpu.dtype.itemsize)
        b_gpu = cuda.mem_alloc(b_cpu.size * b_cpu.dtype.itemsize)
        c_gpu = cuda.mem_alloc(c.size * c.dtype.itemsize)
        cuda.memcpy_htod(a_gpu,a_cpu)
        cuda.memcpy_htod(b_gpu,b_cpu)
        cuda.memcpy_htod(c_gpu,c)
        # Record execution time and execute operation with numpy syntax
        func(a_gpu,b_gpu,c_gpu,np.int32(a_cpu.shape[0]),np.int32(a_cpu.shape[1]), block=block_dim,grid = grid_dim)
        c_naive = np.empty_like(c)
        cuda.memcpy_dtoh(c_naive, c_gpu)
        e_end.record() # wait for event to finish
        pycuda.driver.Context.synchronize()
        
        time_naive = e_start.time_till(e_end)*(1e-3)

        return c_naive, time_naive

    def matrix_mul_optimized1(self, a_cpu, b_cpu):
        """
        Function to perform partially optimized matrix multipication. 
        Should return the result and execution time.
        (Only B tiled in shared memory)
        
        """
        
        func = self.mod1.get_function("Matrix_multiply_optimized1")
        bdim_x = int(math.ceil(a_cpu.shape[0]/2))
        bdim_y = int(math.ceil(a_cpu.shape[1]/2))
        c = np.zeros((a_cpu.shape[0],a_cpu.shape[0]),dtype=np.float32)
        if bdim_x >= bdim_y:
            bdim = bdim_x
        else:
            bdim = bdim_y
        block_dim = (2,2,1)
        grid_dim = (bdim,bdim,1)
        e_start = cuda.Event()
        e_end = cuda.Event()
        e_start.record()
        pycuda.driver.Context.synchronize()
        a_gpu = cuda.mem_alloc(a_cpu.size * a_cpu.dtype.itemsize)
        b_gpu = cuda.mem_alloc(b_cpu.size * b_cpu.dtype.itemsize)
        c_gpu = cuda.mem_alloc(c.size * c.dtype.itemsize)
        cuda.memcpy_htod(a_gpu,a_cpu)
        cuda.memcpy_htod(b_gpu,b_cpu)
        cuda.memcpy_htod(c_gpu,c)
        # Record execution time and execute operation with numpy syntax
        func(a_gpu,b_gpu,c_gpu,np.int32(a_cpu.shape[0]),np.int32(a_cpu.shape[1]), block=block_dim,grid = grid_dim)
        c_optimized1 = np.empty_like(c)
        cuda.memcpy_dtoh(c_optimized1, c_gpu)
        e_end.record() # wait for event to finish
        pycuda.driver.Context.synchronize()
        time_optimized1 = e_start.time_till(e_end)*(1e-3)
        
        return c_optimized1, time_optimized1

    def matrix_mul_optimized2(self, a_cpu, b_cpu):
        """
        Function to perform optimized matrix multiplication using shared
        memory. Should return the result and execution time.
        (A and B both tiled in shared memory)
        
        """
        
        func = self.mod2.get_function("Matrix_multiply_optimized2")
        bdim_x = int(math.ceil(a_cpu.shape[0]/2))
        bdim_y = int(math.ceil(a_cpu.shape[1]/2))
        c = np.zeros((a_cpu.shape[0],a_cpu.shape[0]),dtype=np.float32)
        if bdim_x >= bdim_y:
            bdim = bdim_x
        else:
            bdim = bdim_y
        block_dim = (2,2,1)
        grid_dim = (bdim,bdim,1)
        e_start = cuda.Event()
        e_end = cuda.Event()
        e_start.record()
        pycuda.driver.Context.synchronize()
        a_gpu = cuda.mem_alloc(a_cpu.size * a_cpu.dtype.itemsize)
        b_gpu = cuda.mem_alloc(b_cpu.size * b_cpu.dtype.itemsize)
        c_gpu = cuda.mem_alloc(c.size * c.dtype.itemsize)
        cuda.memcpy_htod(a_gpu,a_cpu)
        cuda.memcpy_htod(b_gpu,b_cpu)
        cuda.memcpy_htod(c_gpu,c)
        # Record execution time and execute operation with numpy syntax
        func(a_gpu,b_gpu,c_gpu,np.int32(a_cpu.shape[0]),np.int32(a_cpu.shape[1]), block=block_dim,grid = grid_dim)
        c_optimized2 = np.empty_like(c)
        cuda.memcpy_dtoh(c_optimized2, c_gpu)
        e_end.record() # wait for event to finish
        pycuda.driver.Context.synchronize()
        time_optimized2 = e_start.time_till(e_end)*(1e-3)
        
        return c_optimized2, time_optimized2 
    
    
    def mul_native_python(self,a_cpu,b_cpu):
        
        start_ = time.time()
        out = np.zeros((a_cpu.shape[0],a_cpu.shape[0]), dtype = np.float32)
        out = np.dot(a_cpu,b_cpu)
        end_ = time.time()
        
        time_py = (end_ - start_)*(1e3)
        
        return out, time_py
        

if __name__ == '__main__':
    # Main code to create arrays, call all functions, calculate average
    # execution and plot
    time_cpu = []
    time_naive = []
    time_opt1 = []
    time_opt2 = []
    M = 2
    N = 3
    for i in range(10):
        a_h = np.random.rand((i+1)*M,(i+1)*N).astype(np.float32)
        b_h = np.random.rand((i+1)*N,(i+1)*M).astype(np.float32)
        mul_func = MatrixMultiply()
        out_cpu, tcpu = mul_func.mul_native_python(a_h,b_h)
        out_naive, tn = mul_func.matrix_mul_naive(a_h,b_h)
        out_opt1,t_opt1 = mul_func.matrix_mul_optimized1(a_h,b_h)
        out_opt2, t_opt2 = mul_func.matrix_mul_optimized2(a_h,b_h)
        D = np.allclose(out_cpu,out_naive)
        B = np.array_equal(out_naive,out_opt1)
        C = np.array_equal(out_naive,out_opt2)
        if (D == True and B == True and C == True):
            time_cpu.append(tcpu)
            time_naive.append(tn)
            time_opt1.append(t_opt1)
            time_opt2.append(t_opt2)
            
        else:
            print("Matrices not equal")
            
    tcp = np.mean(np.array(time_cpu))
    tn = np.mean(np.array(time_naive))
    topt1 = np.mean(np.array(time_opt1))
    topt2 = np.mean(np.array(time_opt2))  
    
    print("Execution time on CPU is: "+str(tcp))
    print("Naive GPU execution time for matrix transpose is: " + str(tn))
    print("GPU execution time for partial tiled tmatrix multiplicaton is: " + str(topt1))
    print("GPU execution time for complete tiled tmatrix multiplicaton is: " + str(topt2))
    
    #Plot execution times for CPU and GPU for matrix transpose task
    x_axis = np.arange(10)
    plt.plot(x_axis,np.log(time_cpu), label ="Execution time on CPU")
    plt.plot(x_axis,np.log(time_naive),label="Naive Execution time on GPU")
    plt.plot(x_axis,np.log(time_opt1),label="Execution time for partial tiled multiplication")
    plt.plot(x_axis,np.log(time_opt2),label="Execution time for complete tiled multiplication")
    
    plt.xlabel("Increasing Array size as M=2*i, N=3*i")
    plt.ylabel("Execution time in (ms)")
    plt.legend()
    plt.savefig('Plots/Fig1_pycuda_1.png')
    print("Plot saved!")
