import unittest
import torch
import torch.nn as nn
from aether.trainer.pcgrad_optimizer import PCGradOptimizerWrapper

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize a single weight vector of size 2
        self.w = nn.Parameter(torch.tensor([1.0, 1.0]))

    def forward(self, x):
        return x * self.w

class TestPCGradOptimizer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        import random
        random.seed(42)

    def test_pcgrad_partially_conflicting(self):
        """
        Test that PCGrad correctly projects partially conflicting gradients.
        """
        model = DummyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        pc_optimizer = PCGradOptimizerWrapper(optimizer)
        
        # We manually construct losses to yield specific gradients
        # Task 1 loss: w1 + w2 => grad1 = [1.0, 1.0]
        loss1 = model.w[0] + model.w[1]
        
        # Task 2 loss: -w1 + 0.5*w2 => grad2 = [-1.0, 0.5]
        loss2 = -model.w[0] + 0.5 * model.w[1]
        
        # Are they conflicting?
        # dot([1,1], [-1, 0.5]) = -1 + 0.5 = -0.5 < 0. Yes, they conflict.
        
        # Expected grad1_pc = grad1 - (dot(grad1, grad2) / dot(grad2, grad2)) * grad2
        # grad1_pc = [1, 1] - (-0.5 / (1.25)) * [-1, 0.5]
        # grad1_pc = [1, 1] + 0.4 * [-1, 0.5] = [0.6, 1.2]
        
        # Expected grad2_pc = grad2 - (dot(grad1, grad2) / dot(grad1, grad1)) * grad1
        # grad2_pc = [-1, 0.5] - (-0.5 / 2.0) * [1, 1]
        # grad2_pc = [-1, 0.5] + 0.25 * [1, 1] = [-0.75, 0.75]
        
        # Expected sum = grad1_pc + grad2_pc = [-0.15, 1.95]
        
        pc_optimizer.pc_backward([loss1, loss2])
        
        expected_grad = torch.tensor([-0.15, 1.95])
        
        self.assertTrue(torch.allclose(model.w.grad, expected_grad, atol=1e-5), 
                        f"Expected {expected_grad}, got {model.w.grad}")

    def test_pcgrad_no_conflict(self):
        """
        Test that PCGrad leaves non-conflicting gradients unchanged (simple sum).
        """
        model = DummyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        pc_optimizer = PCGradOptimizerWrapper(optimizer)
        
        # Task 1 loss: w1 + w2 => grad1 = [1.0, 1.0]
        loss1 = model.w[0] + model.w[1]
        
        # Task 2 loss: 0.5*w1 + 0.5*w2 => grad2 = [0.5, 0.5]
        loss2 = 0.5 * model.w[0] + 0.5 * model.w[1]
        
        # dot([1,1], [0.5, 0.5]) = 1.0 > 0. No conflict.
        # Expected sum = [1.5, 1.5]
        
        pc_optimizer.pc_backward([loss1, loss2])
        
        expected_grad = torch.tensor([1.5, 1.5])
        
        self.assertTrue(torch.allclose(model.w.grad, expected_grad, atol=1e-5),
                        f"Expected {expected_grad}, got {model.w.grad}")

    def test_pcgrad_orthogonality(self):
        """
        Test the Adım 4.1.2 requirement: "Zıt gradyanların birbirini sıfırlamadığını doğrulayan test"
        """
        model = DummyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        pc_optimizer = PCGradOptimizerWrapper(optimizer)
        
        # Task 1: w1 - w2 => grad1 = [1.0, -1.0]
        loss1 = model.w[0] - model.w[1]
        
        # Task 2: -w1 + w2 => grad2 = [-1.0, 1.0]
        loss2 = -model.w[0] + model.w[1]
        
        # They are perfectly opposite. Standard optimizer would sum them to [0, 0].
        # Let's see what PCGrad does.
        # dot = -2
        # grad1_pc = [1, -1] - (-2/2) * [-1, 1] = [1, -1] + [-1, 1] = [0, 0]
        # grad2_pc = [-1, 1] - (-2/2) * [1, -1] = [-1, 1] + [1, -1] = [0, 0]
        # Wait, if they are exactly opposite, they are co-linear and they project to 0.
        
        # To test orthogonality without complete cancellation, we need non-colinear opposing grads.
        # grad1 = [1.0, 0.0]
        # grad2 = [-1.0, 1.0]
        # dot = -1
        # Without PCGrad: sum = [0, 1.0]
        # With PCGrad: 
        # grad1_pc = [1, 0] - (-1/2)*[-1, 1] = [1, 0] + [-0.5, 0.5] = [0.5, 0.5]
        # grad2_pc = [-1, 1] - (-1/1)*[1, 0] = [-1, 1] + [1, 0] = [0, 1]
        # sum = [0.5, 1.5] -> does not cancel out the first dimension completely!
        
        # Reset model
        model = DummyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        pc_optimizer = PCGradOptimizerWrapper(optimizer)
        
        loss1_orth = model.w[0] 
        loss2_orth = -model.w[0] + model.w[1]
        
        pc_optimizer.pc_backward([loss1_orth, loss2_orth])
        
        expected_grad = torch.tensor([0.5, 1.5])
        self.assertTrue(torch.allclose(model.w.grad, expected_grad, atol=1e-5),
                        f"Expected {expected_grad}, got {model.w.grad}")

if __name__ == '__main__':
    unittest.main()
