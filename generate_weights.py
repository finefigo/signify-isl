import numpy as np

# Calculate the total number of weights needed
conv_kernel = np.random.randn(3, 3, 3, 32).astype(np.float32)  # Conv2D kernel
conv_bias = np.random.randn(32).astype(np.float32)  # Conv2D bias
dense_kernel = np.random.randn(5408, 10).astype(np.float32)  # Dense kernel
dense_bias = np.random.randn(10).astype(np.float32)  # Dense bias

# Concatenate all weights into a single array
weights = np.concatenate([
    conv_kernel.flatten(),
    conv_bias.flatten(),
    dense_kernel.flatten(),
    dense_bias.flatten()
])

# Save the weights to a binary file
weights.tofile('model/group1-shard1of1.bin') 