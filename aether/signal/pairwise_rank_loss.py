import torch
import torch.nn as nn

class DateWisePairwiseRankLoss(nn.Module):
    """
    Vektörize PyTorch Broadcasting ($O(K^2)$ Vektörel Hız Matrisi).
    
    741 ikili kıyaslama çiftini (39 x 38 / 2) sıfır for döngüsü ile matrisel olarak hesaplar.
    Margin = 0.10
    L_Rank(A, B) = relu(-sign(Y_A - Y_B) * (y_hat_A - y_hat_B) + margin)
    """
    def __init__(self, margin: float = 0.10):
        super(DateWisePairwiseRankLoss, self).__init__()
        self.margin = margin

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        y_pred: (39,) veya (39, 1) -> Model tahmin skorları
        y_true: (39,) veya (39, 1) -> Risk-Adjusted Target getirileri
        """
        y_pred = y_pred.view(-1, 1)
        y_true = y_true.view(-1, 1)
        
        # 39x39 Matrisel Fark Matrisi (Outer Difference)
        diff_pred = y_pred - y_pred.T  # y_hat_A - y_hat_B
        diff_true = y_true - y_true.T  # Y_A - Y_B
        
        # Sign Matrisi
        target_sign = torch.sign(diff_true)
        
        # Pairwise Marginal Loss Matrisi
        loss_matrix = torch.relu(-target_sign * diff_pred + self.margin)
        
        # Sadece Üst Üçgeni (Köşegen hariç 741 Çifti) al
        triu_mask = torch.triu(torch.ones_like(loss_matrix), diagonal=1).bool()
        loss = loss_matrix[triu_mask].mean()
        
        return loss
