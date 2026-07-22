"""
AetherForecaster-X v35.2 Multi-Asset Master
Katman 3: Valör-Cezalı Bounded Convex QP Solver

Modül: Penalty Matrix Engine (Ceza Matrisleri Motoru)
Faz 2.1: Sermaye büyüklüğüne, piyasa derinliğine ve fon valör kurallarına
göre QP Solver'ı regüle edecek çapraz-maliyet (Penalty) matrislerini üretir.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from aether.data.pit_snapshot import AssetType, DataPoint, PITDataSnapshot
from aether.data.auction_ratio import AuctionRatioResult
from aether.data.hybrid_volatility import VolatilityResult


@dataclass
class PenaltyMatrixResult:
    """
    Hesaplanan kuadratik etki (impact) ve valör (settlement) ceza matrislerini
    taşıyan mühürlü veri yapısı. Tüm matrisler N x N köşegen formattadır.
    """
    symbols: List[str]
    total_capital: float
    adv_effective: Dict[str, float]
    d_impact: pd.DataFrame
    d_settlement: pd.DataFrame
    d_total: pd.DataFrame


class PenaltyMatrixEngine:
    """
    Sermaye ve Valör Cezası Matrisleri Motoru.
    - Hisseler/BYF'ler için kuadratik piyasa etkisi (D_impact)
    - TEFAS fonları için valör ve takas cezası (D_settlement)
    hesaplamalarını tek seferde eksiksiz üretir.
    """

    def __init__(
        self,
        gamma_0: float = 0.01,
        beta_valor: float = 0.001,
        adv_window_days: int = 20,
        epsilon: float = 1e-6
    ):
        """
        :param gamma_0: Likidite etkisi ölçeklendirme katsayısı.
        :param beta_valor: Valör gün (T+N) başına sermaye ceza katsayısı.
        :param adv_window_days: Ortalama ciro hesaplaması için geçmiş gün sayısı (Varsayılan 20).
        :param epsilon: Sıfıra bölme (ZeroDivisionError) zırhı tabanı.
        """
        self.gamma_0 = gamma_0
        self.beta_valor = beta_valor
        self.adv_window_days = adv_window_days
        self.epsilon = epsilon

    def _compute_adv_20(self, points: List[DataPoint]) -> float:
        """
        Geçmiş günlerin TL ciro hacimlerinin (Close * Volume) ortalamasını hesaplar.
        """
        if not points:
            return 0.0

        # DataPoint'leri güne göre grupla ve her günün son fiyatı * son toplam hacmi ciro olarak al.
        # DataPoint içerisinde volume zaten kümülatif olabilir veya 15dk barlardır.
        # Günlük toplam hacmi ve son fiyatı bulalım:
        records = []
        for pt in points:
            records.append({
                "date": pt.timestamp.date(),
                "price": pt.price,
                "volume": pt.volume
            })
            
        df = pd.DataFrame(records)
        if df.empty:
            return 0.0
            
        # Gün bazında toplanmış ciro hesabı: 
        # (15-dk barlarda ciro hesabı için her barın price * volume'u toplanabilir. 
        # Ancak basitçe günlük toplam hacim * kapanış fiyatı da kabul edilebilir.
        # Matematiksel olarak daha hassas olması için bar bazında ciroları (price * volume) toplayalım)
        df["turnover"] = df["price"] * df["volume"]
        daily_turnover = df.groupby("date")["turnover"].sum().reset_index()
        
        daily_turnover.sort_values("date", inplace=True)
        recent_turnovers = daily_turnover["turnover"].tail(self.adv_window_days).values
        
        if len(recent_turnovers) == 0:
            return 0.0
            
        return float(np.mean(recent_turnovers))

    def compute_penalty_matrices(
        self,
        snapshot: PITDataSnapshot,
        auction_ratios: Dict[str, AuctionRatioResult],
        volatilities: Dict[str, VolatilityResult],
        total_capital: float
    ) -> PenaltyMatrixResult:
        """
        D_impact, D_settlement ve D_total N x N köşegen matrislerini üretir.
        """
        # Snapshot'taki tüm sembolleri sıralı şekilde alalım
        symbols = sorted(list(snapshot.get_all_symbols()))
        n = len(symbols)
        
        d_impact_matrix = np.zeros((n, n), dtype=float)
        d_settlement_matrix = np.zeros((n, n), dtype=float)
        adv_effective_dict: Dict[str, float] = {}
        
        if total_capital <= 0:
            raise ValueError(f"Geçersiz sermaye büyüklüğü! total_capital > 0 olmalıdır: {total_capital}")

        for i, sym in enumerate(symbols):
            asset_type = snapshot.get_symbol_asset_type(sym)
            vol_res = volatilities.get(sym)
            auction_res = auction_ratios.get(sym)
            
            # 1. Volatilite ve Alpha değerlerinin alınması (NaN, Inf & None Koruma Zırhı)
            if vol_res is not None and vol_res.volatility is not None and np.isfinite(vol_res.volatility):
                sigma_i = max(float(vol_res.volatility), self.epsilon)
            else:
                sigma_i = self.epsilon
            
            if auction_res is not None and auction_res.alpha_auction is not None and np.isfinite(auction_res.alpha_auction):
                alpha_i = max(float(auction_res.alpha_auction), self.epsilon)
            else:
                alpha_i = 1.0
            
            # 2. Varlık noktalarının alınması ve ADV_20 ciro hesabı
            asset_dict = snapshot.get_data_by_asset_type(asset_type)
            points = asset_dict.get(sym, [])
            
            adv_20 = self._compute_adv_20(points)
            if not np.isfinite(adv_20):
                adv_20 = 0.0
                
            adv_eff = max(adv_20 * alpha_i, self.epsilon)
            adv_effective_dict[sym] = adv_eff
            
            # 3. D_impact (Likidite Ceza Matrisi) Hesabı
            if asset_type in (AssetType.EQUITY, AssetType.ETF):
                impact_penalty = self.gamma_0 * sigma_i * ((total_capital ** 2) / (adv_eff ** 2))
                if not np.isfinite(impact_penalty):
                    impact_penalty = 0.0
                d_impact_matrix[i, i] = impact_penalty
            else:
                # TEFAS Fonları için D_impact = 0.0
                d_impact_matrix[i, i] = 0.0
                
            # 4. D_settlement (Valör Ceza Matrisi) Hesabı
            # Varlığın son noktasındaki settlement_days alınır. Kayıt yoksa varsayılan kurallar uygulanır.
            if points:
                settlement_days = points[-1].settlement_days
            else:
                if asset_type in (AssetType.EQUITY, AssetType.ETF):
                    settlement_days = 2
                elif asset_type == AssetType.TEFAS_LIQUID:
                    settlement_days = 0
                else:
                    settlement_days = 1  # TEFAS_FREE varsayılanı
                
            if settlement_days == 0:
                d_settlement_matrix[i, i] = 0.0
            else:
                d_settlement_matrix[i, i] = self.beta_valor * settlement_days

        # Pandas DataFrame'e dönüştürme
        df_impact = pd.DataFrame(d_impact_matrix, index=symbols, columns=symbols)
        df_settlement = pd.DataFrame(d_settlement_matrix, index=symbols, columns=symbols)
        df_total = df_impact + df_settlement

        return PenaltyMatrixResult(
            symbols=symbols,
            total_capital=total_capital,
            adv_effective=adv_effective_dict,
            d_impact=df_impact,
            d_settlement=df_settlement,
            d_total=df_total
        )

    def compute_penalty_matrices_from_series(
        self,
        volatilities: pd.Series,
        adv_series: pd.Series,
        settlement_days: pd.Series,
        W_total: float,
        symbols: List[str],
        asset_types: Optional[Dict[str, str]] = None
    ) -> PenaltyMatrixResult:
        """
        Doğrudan pandas Series nesneleri kullanarak D_impact, D_settlement ve D_total N x N matrislerini üretir.
        """
        n = len(symbols)
        d_impact_matrix = np.zeros((n, n), dtype=float)
        d_settlement_matrix = np.zeros((n, n), dtype=float)
        adv_effective_dict: Dict[str, float] = {}

        if W_total <= 0:
            raise ValueError(f"Geçersiz sermaye büyüklüğü! W_total > 0 olmalıdır: {W_total}")

        for i, sym in enumerate(symbols):
            sigma_i = max(float(volatilities.get(sym, self.epsilon)), self.epsilon)
            adv_i = float(adv_series.get(sym, 0.0))
            settle_i = int(settlement_days.get(sym, 0))

            atype = "EQUITY"
            if asset_types and sym in asset_types:
                atype = str(asset_types[sym]).upper()
            elif sym.startswith("TEFAS"):
                atype = "TEFAS"

            adv_eff = max(adv_i, self.epsilon)
            adv_effective_dict[sym] = adv_eff

            if atype.startswith("TEFAS"):
                d_impact_matrix[i, i] = 0.0
            else:
                impact_penalty = self.gamma_0 * sigma_i * ((W_total ** 2) / (adv_eff ** 2))
                d_impact_matrix[i, i] = impact_penalty if np.isfinite(impact_penalty) else 0.0

            d_settlement_matrix[i, i] = self.beta_valor * settle_i if settle_i > 0 else 0.0

        df_impact = pd.DataFrame(d_impact_matrix, index=symbols, columns=symbols)
        df_settlement = pd.DataFrame(d_settlement_matrix, index=symbols, columns=symbols)
        df_total = df_impact + df_settlement

        return PenaltyMatrixResult(
            symbols=symbols,
            total_capital=W_total,
            adv_effective=adv_effective_dict,
            d_impact=df_impact,
            d_settlement=df_settlement,
            d_total=df_total
        )

