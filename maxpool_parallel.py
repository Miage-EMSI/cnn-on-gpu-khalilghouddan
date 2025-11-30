import numpy as np
from numba import cuda

class MaxPool2:
    def forward(self, input):
        self.last_input = input
        h, w, num_filters = input.shape
        output = np.zeros((h // 2, w // 2, num_filters), dtype=input.dtype)

        # Define threads per block and blocks per grid
        threads_per_block = (16, 16, 1)
        blocks_per_grid_x = (output.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_per_grid_y = (output.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y, num_filters)

        # Copy data to GPU
        d_input = cuda.to_device(input)
        d_output = cuda.to_device(output)

        # Launch kernel
        maxpool_forward_cuda[blocks_per_grid, threads_per_block](d_input, d_output)

        # Copy result back to host
        d_output.copy_to_host(output)
        return output

    def backprop(self, d_L_d_out):
        h, w, num_filters = self.last_input.shape
        d_L_d_input = np.zeros(self.last_input.shape, dtype=self.last_input.dtype)

        threads_per_block = (16, 16, 1)
        blocks_per_grid_x = (d_L_d_out.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_per_grid_y = (d_L_d_out.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y, num_filters)

        d_last_input = cuda.to_device(self.last_input)
        d_d_L_d_out = cuda.to_device(d_L_d_out)
        d_d_L_d_input = cuda.to_device(d_L_d_input)

        maxpool_backprop_cuda[blocks_per_grid, threads_per_block](d_last_input, d_d_L_d_out, d_d_L_d_input)

        d_d_L_d_input.copy_to_host(d_L_d_input)
        return d_L_d_input


# CUDA kernel for forward pass
@cuda.jit
def maxpool_forward_cuda(input, output):
    i, j, f = cuda.grid(3)
    h, w, num_filters = input.shape
    new_h = h // 2
    new_w = w // 2

    if i < new_h and j < new_w and f < num_filters:
        max_val = input[i*2, j*2, f]
        for ii in range(2):
            for jj in range(2):
                val = input[i*2+ii, j*2+jj, f]
                if val > max_val:
                    max_val = val
        output[i, j, f] = max_val

# CUDA kernel for backward pass
@cuda.jit
def maxpool_backprop_cuda(last_input, d_L_d_out, d_L_d_input):
    i, j, f = cuda.grid(3)
    h, w, num_filters = last_input.shape
    new_h = h // 2
    new_w = w // 2

    if i < new_h and j < new_w and f < num_filters:
        max_val = last_input[i*2, j*2, f]
        max_i = 0
        max_j = 0
        # find max position
        for ii in range(2):
            for jj in range(2):
                val = last_input[i*2+ii, j*2+jj, f]
                if val > max_val:
                    max_val = val
                    max_i = ii
                    max_j = jj
        d_L_d_input[i*2+max_i, j*2+max_j, f] = d_L_d_out[i, j, f]
