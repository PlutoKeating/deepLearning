"""Autograd example adapted from PyTorch Learn the Basics."""

import torch
import torch.nn.functional as F


def main() -> None:
    torch.manual_seed(0)

    print("=== 05 Autograd ===")

    x = torch.ones(5)


    y = torch.zeros(3)
    # requires_grad=True tells autograd to record operations for these tensors.
    w = torch.randn(5, 3, requires_grad=True)
    b = torch.randn(3, requires_grad=True)

    print(f"w.requires_grad: {w.requires_grad}")
    print(f"b.requires_grad: {b.requires_grad}")

    z = torch.matmul(x, w) + b
    # The loss connects model output to the target and becomes the scalar to optimize.
    loss = F.binary_cross_entropy_with_logits(z, y)

    print("\nModel output z:")
    print(z)
    # grad_fn shows that PyTorch built a computation graph for this tensor.
    print(f"z.grad_fn: {z.grad_fn}")
    print(f"loss value: {loss.item():.6f}")
    print(f"loss.grad_fn: {loss.grad_fn}")

    # backward() applies automatic differentiation and fills .grad fields.
    loss.backward()

    print("\nGradient of w after backward():")
    print(w.grad)
    print("\nGradient of b after backward():")
    print(b.grad)


if __name__ == "__main__":
    main()
