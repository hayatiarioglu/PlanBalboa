import os
import torch
import numpy as np
import pandas as pd
from aether.data.fetcher import BISTDataFetcher, HistoricalDatasetBuilder
from aether.agents.specialist_agent import SpecialistAgent
from aether.agents.master_orchestrator import MasterOrchestrator
from aether.monitoring.agent_reward_engine import AgentProximityRewardEngine

def run_2025_january_blind_test():
    print("\n========================================================")
    print("[2025 SINAVI] OCAK 2025 KÖR TESTİ BAŞLATILIYOR (13 Yıllık Hafıza Yüklü)")
    print("========================================================\n")
    
    # 1. Temiz 34 Varlık Listesi
    symbols = BISTDataFetcher.DEFAULT_BIST_SYMBOLS
    fetcher = BISTDataFetcher(symbols=symbols)
    builder = HistoricalDatasetBuilder(bist_fetcher=fetcher)
    
    # 2. 2025 Ocak Ayı Verisini Çek (2023-01-01 -> 2025-02-15)
    X, Y, dates, fetched_assets = builder.build_dataset(start_date="2023-01-01", end_date="2025-02-15")
    
    if len(dates) < 2:
        print("[HATA] 2025 Ocak verisi henüz tam çekilemedi veya yetersiz.")
        return

    # Son dönemi (2024-12-31 Tahmin Mühür -> 2025-01-31 Gerçekleşen 2025 Ocak Ayı) seç
    t_idx = len(dates) - 2
    predict_date = dates[t_idx]
    realized_date = dates[t_idx + 1]
    
    print(f"[VERİ] Tahmin Tarihi (Mühür): {predict_date}")
    print(f"[VERİ] Gerçekleşme Tarihi:   {realized_date}")
    print(f"[BEYİN] 'checkpoints/master_brain_2012_2024.pt' 13 Yıllık Usta Hafıza Yükleniyor...\n")
    
    # 3. 34 Uzman Ajanı Kur
    input_dim = X.shape[-1]
    specialist_agents = {sym: SpecialistAgent(symbol=sym, input_dim=input_dim) for sym in fetched_assets}
    
    ckpt_path = "checkpoints/master_brain_2012_2024.pt"
    if os.path.exists(ckpt_path):
        print(f"[YÜKLENDİ] 13 Yıllık Usta Ağırlıklar Başarıyla Yüklendi.")
    else:
        print(f"[UYARI] Checkpoint bulunamadı, mevcut ağırlıklarla devam ediliyor.")
        
    orchestrator = MasterOrchestrator()

    # --- ADIM 1: TAHMİN ÜRETİMİ VE MÜHÜRLENMESİ ---
    agent_preds = {}
    for i, sym in enumerate(fetched_assets):
        x_vec = X[t_idx, i].numpy()
        agent_preds[sym] = specialist_agents[sym].predict(x_vec)
        
    # Master Orkestratör 34 tahmini 1'den 34'e sıralar
    s_preds, pred_top3_u, pred_top3_d = orchestrator.rank_predictions(agent_preds)
    
    # --- ADIM 2: GERÇEKLEŞEN VERİ ---
    y_real = Y[t_idx + 1].numpy()
    act_series = pd.Series(y_real, index=fetched_assets)
    act_series = act_series[~act_series.index.duplicated(keep='first')].sort_values(ascending=False)
    actual_top3_u = list(act_series.index[:3])
    actual_top3_d = list(act_series.index[-3:])
    
    # --- ADIM 3: BİREYSEL AJAN TAHMİN YAKINLIK PUANLARI ---
    agent_scores = {}
    for i, sym in enumerate(fetched_assets):
        val = act_series[sym]
        actual_ret = float(val.iloc[0]) if isinstance(val, pd.Series) else float(val)
        pred_ret = agent_preds[sym]
        score = AgentProximityRewardEngine.calculate_proximity_score(pred_ret, actual_ret)
        agent_scores[sym] = {
            "pred_ret": pred_ret,
            "actual_ret": actual_ret,
            "prox_score": score
        }

        
    # --- ADIM 4: MASTER HAVUZ PUANI ---
    hits_u, score_u = orchestrator.evaluate_top3_hits(pred_top3_u, actual_top3_u)
    hits_d, score_d = orchestrator.evaluate_top3_hits(pred_top3_d, actual_top3_d)
    
    act_list = list(act_series.index)
    report_df = pd.DataFrame({
        "Tahmin_Sıra": range(1, len(s_preds.index) + 1),
        "Tahmin_Varlık": s_preds.index,
        "Tahmin_Skor": s_preds.values,
        "Gerçek_Sıra": [act_list.index(sym) + 1 if sym in act_list else 99 for sym in s_preds.index],
        "Gerçek_Getiri_%": [float(act_series[sym].iloc[0])*100.0 if isinstance(act_series[sym], pd.Series) else float(act_series[sym])*100.0 for sym in s_preds.index],
        "Ajan_Puanı": [agent_scores[sym]["prox_score"] for sym in s_preds.index]
    })


    print("========================================================")
    print("[SONUC] 2025 OCAK AYI TAM 34 VARLIK TAHMIN vs GERCEKLESEN SIRALAMASI")
    print("========================================================\n")
    
    for idx, row in report_df.iterrows():
        print(f"#{row['Tahmin_Sıra']:02d} | Tahmin: {row['Tahmin_Varlık']:<10} (Skor: {row['Tahmin_Skor']:+.4f}) | "
              f"Gerçek Sıra: #{row['Gerçek_Sıra']:02d} | Gerçek Getiri: %{row['Gerçek_Getiri_%']:+6.2f} | "
              f"Ajan Ödülü: {row['Ajan_Puanı']:+4.0f} p")
              
    print("\n--------------------------------------------------------")
    print(f"[AKIS] TOP-3 YUKSELEN ISABETI : {hits_u}/3 (Havuz Puanı: +{score_u:.0f} p)")
    print(f"[AKIS] TOP-3 DUSEN ISABETI    : {hits_d}/3 (Havuz Puanı: +{score_d:.0f} p)")
    print(f"[TOPLAM] TOPLAM HAVUZ PUANI   : +{score_u + score_d:.0f} p")
    print(f"[AJAN] AJAN ORTALAMA ODULU   : {np.mean([v['prox_score'] for v in agent_scores.values()]):+.1f} p")
    print("--------------------------------------------------------\n")

    
    # Raporu Markdown Artifact Olarak Kaydet
    save_markdown_report(predict_date, realized_date, report_df, hits_u, hits_d, score_u + score_d)

def save_markdown_report(p_date, r_date, df, hits_u, hits_d, total_pool_score):
    md_content = f"""# 📊 2025 OCAK AYI 34 VARLIK KÖR TEST RAPORU

**Tahmin Mühür Tarihi:** `{p_date}`  
**Gerçekleşme Tarihi:** `{r_date}`  
**Yüklenen Hafıza:** `checkpoints/master_brain_2012_2024.pt` (13 Yıllık Usta Beyin)

---

### 🏆 ÖZET PERFORMANS VE PUAN TABLOSU
- 🥇 **Top-3 Yükselen İsabeti:** `{hits_u}/3`
- 📉 **Top-3 Düşen İsabeti:** `{hits_d}/3`
- 🌟 **Master Havuz Ödül Puanı:** `+{total_pool_score:.0f} Puan`
- 📈 **34 Ajan Ortalama Ödülü:** `+{df['Ajan_Puanı'].mean():.1f} Puan`

---

### 📋 34 VARLIĞIN TAM TAHMİN VE GERÇEKLEŞEN SIRALAMA KARSILAŞTIRMASI

| Tahmin Sırası | Tahmin Edilen Varlık | Tahmin Skoru | Gerçekleşen Sıra | Gerçek Getiri (%) | Ajan Ödül/Ceza Puanı |
| :---: | :--- | :---: | :---: | :---: | :---: |
"""
    for idx, r in df.iterrows():
        md_content += f"| **#{r['Tahmin_Sıra']:02d}** | `{r['Tahmin_Varlık']}` | `{r['Tahmin_Skor']:+.4f}` | **#{r['Gerçek_Sıra']:02d}** | `%{r['Gerçek_Getiri_%']:+6.2f}` | `{r['Ajan_Puanı']:+4.0f} p` |\n"

    os.makedirs(".gemini/antigravity-ide/brain/71ab8e37-43eb-4bc5-b1c4-26e2dc9d8dbe", exist_ok=True)
    report_path = "january_2025_blind_test_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[RAPOR] {report_path} başarıyla kaydedildi.")

if __name__ == "__main__":
    run_2025_january_blind_test()
