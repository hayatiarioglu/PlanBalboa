import torch
import torch.nn as nn
from typing import List, Tuple, Iterable
import random

class PCGradOptimizerWrapper:
    """
    Projecting Conflicting Gradients (PCGrad) Optimizer Wrapper.
    Wraps an existing PyTorch optimizer. Prevents multi-task gradients from cancelling each other out.
    If grad_i and grad_j conflict (dot product < 0), projects grad_i onto the normal of grad_j.
    """
    
    def __init__(self, optimizer: torch.optim.Optimizer):
        """
        Args:
            optimizer: A standard PyTorch optimizer (e.g., Adam, SGD).
        """
        self.optimizer = optimizer
        
    def zero_grad(self):
        """Clears the gradients of all optimized torch.Tensor s."""
        self.optimizer.zero_grad()
        
    def step(self):
        """Performs a single optimization step using the wrapped optimizer."""
        self.optimizer.step()
        
    def pc_backward(self, losses: Iterable[torch.Tensor]):
        """
        Calculates the gradients for each loss and projects conflicting gradients.
        Args:
            losses: An iterable of scalar loss tensors (one for each task).
        """
        losses = list(losses)
        assert len(losses) > 0, "Must provide at least one loss."
        
        # Step 1: Compute and store gradients for each task independently.
        task_grads = []
        for i, loss in enumerate(losses):
            self.zero_grad()
            
            # Memory Optimization: Only retain graph if it's not the last loss
            is_last = (i == len(losses) - 1)
            loss.backward(retain_graph=not is_last)
            
            # Pack gradients into a single flat tensor for easier vector operations
            grad_vector = self._pack_gradients()
            task_grads.append(grad_vector)
            
        # Clear grads one last time before accumulating the projected ones
        self.zero_grad()
        
        # Step 2: Apply PCGrad algorithm
        # For each task, check conflicts with other tasks in a random order
        projected_grads = []
        
        task_indices = list(range(len(task_grads)))
        
        for i in task_indices:
            grad_i = task_grads[i].clone()
            
            other_indices = [j for j in task_indices if j != i]
            random.shuffle(other_indices)
            
            for j in other_indices:
                grad_j = task_grads[j]
                
                dot_product = torch.dot(grad_i, grad_j)
                if dot_product < 0:
                    # Conflicting gradients. Project grad_i onto the normal of grad_j
                    grad_j_norm_sq = torch.dot(grad_j, grad_j) + 1e-8 # Epsilon protection
                    grad_i = grad_i - (dot_product / grad_j_norm_sq) * grad_j
            
            projected_grads.append(grad_i)
            
        # Step 3: Sum all projected gradients
        if not projected_grads:
            return
            
        total_grad = torch.sum(torch.stack(projected_grads), dim=0)
        
        # Unpack the total gradient back to the model parameters
        self._unpack_gradients(total_grad)
        
    def _pack_gradients(self) -> torch.Tensor:
        """Flattens all gradients of the optimized parameters into a single 1D tensor."""
        grads = []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    if p.grad is not None:
                        grads.append(p.grad.detach().flatten())
                    else:
                        # If requires_grad but grad is None, use a zero tensor
                        grads.append(torch.zeros_like(p).flatten())
        
        if not grads:
            return torch.tensor([])
            
        packed = torch.cat(grads)
        return torch.nan_to_num(packed, nan=0.0, posinf=1.0, neginf=-1.0)
        
    def _unpack_gradients(self, flat_grad: torch.Tensor):
        """Restores the flattened gradient vector back into the param.grad attributes."""
        if flat_grad.numel() == 0:
            return
            
        offset = 0
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    num_param = p.numel()
                    grad_slice = flat_grad[offset : offset + num_param]
                    
                    if p.grad is not None:
                        p.grad.copy_(grad_slice.view_as(p))
                    else:
                        p.grad = grad_slice.view_as(p).clone()
                        
                    offset += num_param
