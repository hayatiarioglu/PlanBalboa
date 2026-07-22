import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

# Path Ekleme
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aether.data.rolling_fetcher import DailyRollingFetcher
from aether.signal.extreme_triplet_loss import ExtremeTripletRankLoss
from aether.models.moe_engine import MoESpecialistNetwork
from aether.agents.master_ranker import MasterRanker

class ExtremeTailQuantTrainer:
    """
    3/3 EXTREME TAIL ENGINE: 100-Epoch MoE + Target-Indexed Triplet Loss + Uncertainty Gate.
    
    Zırhlar & Mayın Kalkanları:
    1. Target-Indexed Triplet Loss: Autograd kopması imha edildi (y_true argsort indeksleme).
    2. MoE Load-Balancing Loss (lambda_aux = 0.10): Expert Collapse (Uzman Çökmesi) sıfırlandı.
    3. Asset-Class Specific Z-Score: Fon vs Hisse Z-score havuzları ayrıştırıldı.
    4. Epistemic Uncertainty Gate (Monte Carlo Dropout Ensemble %70+ Güven Kalkanı).
    """

    def __init__(self, start_date: str = "2012-01-01", end_date: str = "2024-12-31", epochs: int = 100, lr: float = 1e-3):
        self.start_date = start_date
        self.end_date = end_date
        self.epochs = epochs
        self.lr = lr
        
        self.rolling_builder = DailyRollingFetcher()
        self.rank_loss_fn = ExtremeTripletRankLoss(margin=0.10, triplet_margin=0.20, triplet_weight=1.0)
        self.ranker = MasterRanker(confidence_threshold=0.70)

    def train_and_evaluate(self):
        print(f"\n========================================================", flush=True)
        print(f"[3/3 EXTREME TAIL ENGINE] MAYINSIZ 100-EPOCH MOE + TRIPLET EĞİTİMİ", flush=True)
        print(f"Zırhlar: Target-Indexed Triplet + MoE Load Balancing + Uncertainty Gate", flush=True)
        print(f"========================================================\n", flush=True)
        
        # 0. Eski Zihni (Checkpoint) Tamamen Temizle / Sil
        ckpt_path = "checkpoints/master_brain_2012_2024.pt"
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
            print(f"[ZİHİN TEMİZLENDİ] Eski '{ckpt_path}' silindi! Taze zihin ilklendiriliyor...", flush=True)
            
        # 1. Kaydırmalı Veri Setini İnşa Et
        X_all, Y_all, dates, assets = self.rolling_builder.build_rolling_dataset(
            start_date=self.start_date, 
            end_date=self.end_date
        )
        
        num_samples = X_all.shape[0]
        n_assets = len(assets)
        input_dim = X_all.shape[-1]
        
        print(f"[DATA OK] {num_samples} Günlük Snapshot Yüklendi ({n_assets} Varlık, {input_dim} Faktör).", flush=True)
        
        # 2. Train / Val / Blind Test Ayrımı & Purged Embargo
        split_test = num_samples - 252       # Son 1 Yıl (2024) Blind Test
        split_val = split_test - 252         # 2023 Yılı Validation
        embargo_gap = 21
        
        X_train = X_all[:split_val - embargo_gap]
        Y_train = Y_all[:split_val - embargo_gap]
        
        X_val = X_all[split_val:split_test - embargo_gap]
        Y_val = Y_all[split_val:split_test - embargo_gap]
        
        X_test = X_all[split_test:]
        Y_test = Y_all[split_test:]
        test_dates = dates[split_test:]
        
        print(f"[SPLIT] Train: {X_train.shape[0]} Gün | Val: {X_val.shape[0]} Gün | Embargo: {embargo_gap} Gün | Blind Test: {X_test.shape[0]} Gün", flush=True)
        
        # 3. Mixture of Experts (MoE) Ağ Mimarisi
        model = MoESpecialistNetwork(input_dim=input_dim, hidden_dim=64, num_experts=3)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-5)
        
        best_val_loss = float("inf")
        patience = 15
        patience_counter = 0
        best_weights = None
        
        print(f"\n--- 100 EPOCH EXTREME TAIL MOE EĞİTİMİ BAŞLIYOR ---\n", flush=True)
        
        # 4. 100 EPOCH DERİN MOE EĞİTİM DÖNGÜSÜ
        for epoch in range(1, self.epochs + 1):
            model.train()
            epoch_train_loss = 0.0
            perm = torch.randperm(X_train.shape[0])
            
            for idx in perm:
                x_day = X_train[idx] # [39, 5]
                y_day = Y_train[idx] # [39]
                
                # MoE Forward Pass (Output + Load Balancing Aux Loss)
                preds, aux_loss = model(x_day)
                preds = preds.squeeze(-1) # [39]
                
                # Target-Indexed Triplet Loss + Pairwise Loss
                rank_loss = self.rank_loss_fn(preds, y_day)
                
                # Toplam Kayıp: Rank Loss + 0.10 * MoE Load Balancing Aux Loss
                total_loss = rank_loss + 0.10 * aux_loss
                
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_train_loss += total_loss.item()
                
            scheduler.step()
            avg_train_loss = epoch_train_loss / X_train.shape[0]
            
            # VALIDATION PHASE
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for idx in range(X_val.shape[0]):
                    x_val_day = X_val[idx]
                    y_val_day = Y_val[idx]
                    preds_val, aux_val = model(x_val_day)
                    preds_val = preds_val.squeeze(-1)
                    val_loss += (self.rank_loss_fn(preds_val, y_val_day) + 0.10 * aux_val).item()
                    
            avg_val_loss = val_loss / X_val.shape[0]
            
            # EARLY STOPPING CHECK
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_weights = model.state_dict()
            else:
                patience_counter += 1
                
            print(f"  [EPOCH #{epoch:03d}/{self.epochs}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Best Val: {best_val_loss:.6f} | Patience: {patience_counter}/{patience}", flush=True)
                
            if patience_counter >= patience:
                print(f"\n[EARLY STOPPING] Validation Loss {patience} Epoch Boyunca İyileşmedi. Eğitim #{epoch}. Epoch'ta Erken Kesildi!", flush=True)
                break
                
        # En iyi ağırlıkları yükle ve mühürle
        if best_weights is not None:
            model.load_state_dict(best_weights)
                
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        print(f"\n[MÜHÜRLENDİ] Yeni Taze MoE Ağırlıkları '{ckpt_path}' Olarak Kaydedildi.", flush=True)
        
        # 5. BLIND TEST WITH EPISTEMIC UNCERTAINTY GATE (2024 SINAVI)
        print(f"\n========================================================", flush=True)
        print(f"[BLIND TEST] 2024 PİYASA SINAVI (EPISTEMIC UNCERTAINTY GATE)", flush=True)
        print(f"========================================================\n", flush=True)
        
        test_results = []
        for t_idx in range(0, X_test.shape[0] - 21, 21):
            t_date = test_dates[t_idx]
            x_test_day = X_test[t_idx]
            y_test_real = Y_test[t_idx + 21].numpy()
            
            # Monte Carlo Dropout Uncertainty Gate Eval
            pred_top3_u, pred_top3_d, consensus, is_confident = self.ranker.evaluate_uncertainty_gate(
                model=model, 
                x_day=x_test_day, 
                assets=assets, 
                num_passes=10
            )
            
            act_series = pd.Series(y_test_real, index=assets)
            actual_top3_u = list(act_series.nlargest(3).index)
            actual_top3_d = list(act_series.nsmallest(3).index)
            
            hits_u = len(set(pred_top3_u).intersection(set(actual_top3_u)))
            hits_d = len(set(pred_top3_d).intersection(set(actual_top3_d)))
            
            score_u = 100.0 if hits_u == 3 else (50.0 if hits_u == 2 else (10.0 if hits_u == 1 else 0.0))
            score_d = 100.0 if hits_d == 3 else (50.0 if hits_d == 2 else (10.0 if hits_d == 1 else 0.0))
            total_score = score_u + score_d
            
            trade_status = "İŞLEM AÇILDI" if is_confident else "PAS GEÇİLDİ (BEKLE)"
            
            test_results.append({
                "date": t_date,
                "hits_up": hits_u,
                "hits_down": hits_d,
                "score": total_score,
                "consensus": consensus,
                "is_confident": is_confident,
                "pred_up": ", ".join(pred_top3_u),
                "actual_up": ", ".join(actual_top3_u),
                "pred_down": ", ".join(pred_top3_d),
                "actual_down": ", ".join(actual_top3_d)
            })
            
            print(f"[BLIND TEST] [{t_date}] Karar: {trade_status} (Güven: %{consensus*100:.1f}) | Puan: +{total_score:.0f}\n"
                  f"  [YÜKSELEN] Tahmin: [{', '.join(pred_top3_u)}] vs Gerçek: [{', '.join(actual_top3_u)}]\n"
                  f"  [DÜŞEN]    Tahmin: [{', '.join(pred_top3_d)}] vs Gerçek: [{', '.join(actual_top3_d)}]", flush=True)
            
        df_test = pd.DataFrame(test_results)
        df_active = df_test[df_test["is_confident"]]
        
        print(f"\n========================================================", flush=True)
        print(f"[NİHAİ BAŞARI RAPORU] 3/3 EXTREME TAIL ENGINE 2024 BLIND TEST", flush=True)
        print(f"Toplam Dönem Sayısı        : {len(df_test)} Ay", flush=True)
        print(f"Aktif İşlem Açılan Dönem   : {len(df_active)} Ay", flush=True)
        if len(df_active) > 0:
            print(f"Aktif Dönem Ortalama Puanı : +{df_active['score'].mean():.1f} / 200 Puan", flush=True)
            print(f"Aktif Yükselen İsabeti     : {df_active['hits_up'].mean():.2f} / 3", flush=True)
            print(f"Aktif Düşen İsabeti       : {df_active['hits_down'].mean():.2f} / 3", flush=True)
        print(f"Tüm Dönemler Ort. Puanı    : +{df_test['score'].mean():.1f} / 200 Puan", flush=True)
        print(f"========================================================\n", flush=True)
        
        return df_test

if __name__ == "__main__":
    trainer = ExtremeTailQuantTrainer(epochs=100)
    trainer.train_and_evaluate()
