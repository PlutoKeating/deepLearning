"""Tensor examples adapted from PyTorch Learn the Basics: Tensors."""

import numpy as np
import torch


def main() -> None:
    # Fixing the seed makes the random tensor values stable for screenshots.
    torch.manual_seed(0)

    print("=== 01 Tensors ===")

    # torch.tensor creates a tensor by copying regular Python data.
    data = [[1, 2], [3, 4]]
    x_data = torch.tensor(data)
    print("\nTensor from data:")
    print(x_data)

    # torch.from_numpy shares data with a NumPy array, matching the basics tutorial.
    np_array = np.array(data)
    x_np = torch.from_numpy(np_array)
    print("\nTensor from NumPy array:")
    print(x_np)

    # *_like functions reuse the shape and device of an existing tensor.
    x_ones = torch.ones_like(x_data)
    print("\nOnes tensor with same shape:")
    print(x_ones)

    # dtype can be changed when making a new tensor with the same shape.
    x_rand = torch.rand_like(x_data, dtype=torch.float)
    print("\nRandom tensor with same shape and float dtype:")
    print(x_rand)

    # Shape, dtype, and device are the three key pieces of tensor metadata.
    tensor = torch.rand(3, 4)
    print("\nTensor metadata:")
    print(f"Shape of tensor: {tensor.shape}")
    print(f"Datatype of tensor: {tensor.dtype}")
    print(f"Device tensor is stored on: {tensor.device}")

    # Moving tensors to CUDA is optional; later labs keep CPU execution valid.
    if torch.cuda.is_available():
        tensor = tensor.to("cuda")
        print(f"\nTensor moved to CUDA device: {tensor.device}")
    else:
        print("\nCUDA is not available; tensor stays on CPU.")

    tensor = torch.ones(4, 4)
    # Tensor indexing works similarly to NumPy indexing.
    print("\nFirst row:")
    print(tensor[0])
    print("First column:")
    print(tensor[:, 0])
    print("Last column before assignment:")
    print(tensor[..., -1])
    tensor[:, 1] = 0
    print("Tensor after setting column 1 to zero:")
    print(tensor)

    # Concatenation joins tensors along a selected dimension.
    t1 = torch.cat([tensor, tensor, tensor], dim=1)
    print("\nConcatenated tensor shape:")
    print(t1.shape)

    # Element-wise multiplication keeps the same shape.
    print("\nElement-wise multiplication:")
    print(tensor.mul(tensor))

    # Matrix multiplication follows linear algebra shape rules.
    print("\nMatrix multiplication:")
    print(tensor.matmul(tensor.T))

    # Operations ending with "_" modify the tensor in place.
    tensor.add_(5)
    print("\nIn-place add 5:")
    print(tensor)


if __name__ == "__main__":
    main()
