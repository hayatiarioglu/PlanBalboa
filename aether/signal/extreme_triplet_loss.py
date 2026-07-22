import torch
import torch.nn as nn

class ExtremeTripletRankLoss(nn.Module):
    """
    3/3 Extreme Tail Engine Target-Indexed Türevlenebilir Triplet Rank Loss.
    
    Zırhlar:
    1. Pairwise Marginal Loss: Tüm ikili çiftler (741 çift) arasında matrisel sıralama kaybı.
    2. Target-Indexed Triplet Loss:
       İndeksler PyTorch sort ile DEĞİL, GERÇEK HEDEF GETİRİLERE (Y_true) göre sabitlenir.
       Top-3 ve Bottom-3 varlıkların tahmin edilen skorları ortadaki 4. sıradaki varlıktan devasa bir marjin (alpha=0.20) ile ayrılır.
    3. Autograd grafik kopması tamamen engellenmiştir.
    """
    def __init__(self, margin: float = 0.10, triplet_margin: float = 0.20, triplet_weight: float = 1.0):
        super(ExtremeTripletRankLoss, self).__init__()
        self.margin = margin
        self.triplet_margin = triplet_margin
        self.triplet_weight = triplet_weight

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        y_pred: (N,) - Tahmin edilen risk-adjusted Z-skorları
        y_true: (N,) - Gerçekleşen kesitsel Z-skorları
        """
        N = y_pred.shape[0]
        if N < 6:
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)
            
        # 1. Matrisel Vektörize Pairwise Rank Loss (O(N^2))
        diff_pred = y_pred.unsqueeze(1) - y_pred.unsqueeze(0) # (N, N)
        diff_true = y_true.unsqueeze(1) - y_true.unsqueeze(0) # (N, N)
        
        target_sign = torch.sign(diff_true)
        pairwise_loss = torch.relu(-target_sign * diff_pred + self.margin)
        
        mask = torch.triu(torch.ones(N, N, device=y_pred.device), diagonal=1).bool()
        loss_pairwise = pairwise_loss[mask].mean()
        
        # 2. TARGET-INDEXED TRIPLET LOSS (AUTOGRAD KORUMALI)
        # İndeksler y_true'ya göre sabitlenir (PyTorch sort ile gradyan kopmaz!)
        sorted_indices = torch.argsort(y_true)
        
        idx_top3 = sorted_indices[-3:]      # Gerçek Top 3 varlık indeksi
        idx_top4th = sorted_indices[-4]     # Gerçek 4. varlık indeksi
        
        idx_bottom3 = sorted_indices[:3]    # Gerçek Bottom 3 varlık indeksi
        idx_bottom4th = sorted_indices[3]   # Gerçek 4. varlık indeksi
        
        # Top-3 Tahminleri vs 4. Varlık Tahmini
        pred_top3_min = y_pred[idx_top3].min()
        pred_4th = y_pred[idx_top4th]
        triplet_top_loss = torch.relu(pred_4th - pred_top3_min + self.triplet_margin)
        
        # Bottom-3 Tahminleri vs 4. Varlık Tahmini
        pred_bottom3_max = y_pred[idx_bottom3].max()
        pred_bottom4th = y_pred[idx_bottom4th]
        triplet_bottom_loss = torch.relu(pred_bottom3_max - pred_bottom4th + self.triplet_margin)
        
        total_loss = loss_pairwise + self.triplet_weight * (triplet_top_loss + triplet_bottom_loss)
        return total_loss
