"""
AetherForecaster-X v35.2 Multi-Asset Master
Katman 3: Valör-Cezalı Bounded Convex QP Solver

Modül: Portfolio QP Matrix Builder & Constraint Assembler (Matris Birleştirici ve Kısıt Motoru)
Faz 2.4 / Adım 2.4.3: Eşitsizlik Kısıtlarının (Inequality Constraints) tanımlanması:
0 <= w <= w_max (Long-only ve Varlık Başına Maks. Ağırlık Zırhı).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


@dataclass
class GroupConstraint:
    """
    Belirli bir varlık grubu için maruziyet (exposure) kısıtı.
    Örn: TEFAS Likit Fonlar toplamı >= 0.05, <= 0.20
    """
    group_name: str
    symbols: List[str]
    min_weight: float = 0.0
    max_weight: float = 1.0


@dataclass
class PortfolioQPConfig:
    """
    Portföy optimizasyonu kısıt yapılandırması.
    """
    target_total_weight: float = 1.0                  # Bütçe eşitlik kısıtı hedefi (Sum x = 1.0)
    min_asset_weight: float = 0.0                     # Varsayılan varlık bazlı alt sınır (Long-only: 0.0)
    max_asset_weight: float = 1.0                     # Varsayılan varlık bazlı üst sınır (Tek hisse maks: %100 veya %20)
    allow_short: bool = False                         # Açığa satıp kısa pozisyona izin verilip verilmediği
    asset_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict) # Sembol bazlı özel (min, max) sınırları
    group_constraints: List[GroupConstraint] = field(default_factory=list)      # Grup maruziyet kısıtları
    w_prev: Optional[Dict[str, float]] = None          # Faz 2.5.1: Önceki portföy ağırlıkları w_{t-1}
    max_total_turnover: Optional[float] = None         # Faz 2.5.1: Toplam devir hızı üst sınırı (Sum u_i <= max_turnover)
    max_asset_turnover: Dict[str, float] = field(default_factory=dict) # Faz 2.5.1: Varlık bazlı maks pozisyon değişimi (u_i <= max_change)


@dataclass
class QPProblemInput:
    """
    OSQP Çözücüye beslenecek eksiksiz matris ve vektör paketi.
    """
    P: np.ndarray                 # Koşullanmış Hessian Matrisi (N x N veya 2N x 2N)
    q: np.ndarray                 # Lineer maliyet / Alpha sinyal vektörü (-mu, N veya 2N)
    A: np.ndarray                 # Kısıt Matrisi (M x N veya M x 2N)
    l: np.ndarray                 # Kısıt Alt Sınırları (M,)
    u: np.ndarray                 # Kısıt Üst Sınırları (M,)
    symbols: List[str]            # Varlık sembollerinin sıralı listesi (N,)
    constraint_names: List[str]   # Kısıt satırlarının açıklayıcı isimleri (M,)
    inequality_mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool)) # True if row is an inequality constraint
    is_turnover_expanded: bool = False # True if decision vector expanded to [w, u] (2N)


class PortfolioQPBuilder:
    """
    Hessian (P) matrisi ve Alpha sinyalini (q = -mu) bütçe kısıtları, kutusal varlık sınırları,
    grup maruziyet kısıtları ve turn-over (pozisyon değişim) kısıtlarıyla birleştirerek OSQP için
    (P, q, A, l, u) matrislerini oluşturan motor.
    """

    def validate_inequality_bounds(self, config: PortfolioQPConfig, symbols: List[str]) -> None:
        """
        Adım 2.4.3 & 2.5.1: Eşitsizlik kısıtlarının (Inequality Constraints: 0 <= w <= w_max ve Turn-over)
        mantıksal ve matematiksel geçerliliğini denetler.
        """
        total_max_possible = 0.0

        for sym in symbols:
            min_w, max_w = config.asset_bounds.get(sym, (config.min_asset_weight, config.max_asset_weight))

            # Long-only denetimi
            if not config.allow_short and min_w < 0.0:
                raise ValueError(f"Long-only portföyde {sym} için alt sınır ({min_w}) 0'dan küçük olamaz!")

            if min_w > max_w:
                raise ValueError(f"{sym} için alt sınır ({min_w}) üst sınırdan ({max_w}) büyük olamaz!")

            if max_w < 0.0:
                raise ValueError(f"{sym} için üst sınır ({max_w}) negatif olamaz!")

            total_max_possible += max_w

        # Çözümsüzlük (Infeasibility) denetimi: Sum(w_max) < target_total_weight ise bütçe sağlamaz!
        if total_max_possible < config.target_total_weight - 1e-8:
            raise ValueError(
                f"Çözümsüzlük Uyarısı! Maksimum varlık ağırlıklarının toplamı ({total_max_possible:.4f}), "
                f"hedef bütçeden ({config.target_total_weight:.4f}) küçük! Optimizasyon imkansız."
            )

        if config.max_total_turnover is not None and config.max_total_turnover < 0.0:
            raise ValueError(f"Maksimum toplam devir hızı sınırı ({config.max_total_turnover}) negatif olamaz!")

    def build_qp_input(
        self,
        P_conditioned: Union[np.ndarray, pd.DataFrame],
        mu_norm: Union[np.ndarray, pd.Series],
        symbols: List[str],
        config: Optional[PortfolioQPConfig] = None
    ) -> QPProblemInput:
        """
        Girdileri alır, bütçe eşitliğini, kutusal varlık sınırlarını ve Faz 2.5.1 Turn-over kısıtlarını inşa eder.

        :param P_conditioned: HessianConditioningEngine çıktısı N x N matris.
        :param mu_norm: AlphaSignalEngine çıktısı N boyutlu vektör.
        :param symbols: N adet sembol listesi (Matris dizilim sırasını belirler).
        :param config: Kısıt konfigürasyonu.
        """
        if config is None:
            config = PortfolioQPConfig()

        # 1. Sembol ve Boyut Kontrolleri
        n_assets = len(symbols)
        if n_assets == 0:
            raise ValueError("Varlık sembol listesi boş olamaz!")

        self.validate_inequality_bounds(config, symbols)

        if isinstance(P_conditioned, pd.DataFrame):
            P_arr = P_conditioned.loc[symbols, symbols].to_numpy(dtype=np.float64)
        else:
            P_arr = np.asarray(P_conditioned, dtype=np.float64)

        if P_arr.shape != (n_assets, n_assets):
            raise ValueError(f"P_conditioned matrisi boyutu {P_arr.shape}, sembol sayısı ({n_assets}) ile uyuşmuyor!")

        if isinstance(mu_norm, pd.Series):
            mu_arr = mu_norm.reindex(symbols).to_numpy(dtype=np.float64)
        else:
            mu_arr = np.asarray(mu_norm, dtype=np.float64).flatten()

        if len(mu_arr) != n_assets:
            raise ValueError(f"mu_norm boyutu ({len(mu_arr)}), sembol sayısı ({n_assets}) ile uyuşmuyor!")

        if not (np.all(np.isfinite(P_arr)) and np.all(np.isfinite(mu_arr))):
            raise ValueError("P_conditioned veya mu_norm verisinde NaN/Inf tespit edildi!")

        q_arr = -mu_arr

        # Turn-over kısıtı varsa karar değişkeni boyutu 2N olur (x = [w, u])
        has_turnover = config.w_prev is not None
        n_vars = 2 * n_assets if has_turnover else n_assets

        if has_turnover:
            P_full = np.zeros((2 * n_assets, 2 * n_assets), dtype=np.float64)
            P_full[:n_assets, :n_assets] = P_arr
            q_full = np.zeros(2 * n_assets, dtype=np.float64)
            q_full[:n_assets] = q_arr
        else:
            P_full = P_arr
            q_full = q_arr

        # 2. Kısıt Satırlarının İnşası (A, l, u)
        A_rows: List[np.ndarray] = []
        l_rows: List[float] = []
        u_rows: List[float] = []
        constraint_names: List[str] = []
        ineq_mask: List[bool] = []

        # A) BÜTÇE EŞİTLİK KISITI: Sum(w_i) = target_total_weight
        budget_row = np.zeros(n_vars, dtype=np.float64)
        budget_row[:n_assets] = 1.0
        A_rows.append(budget_row)
        l_rows.append(float(config.target_total_weight))
        u_rows.append(float(config.target_total_weight))
        constraint_names.append("Budget_Equality_Sum_w")
        ineq_mask.append(False)

        # B) VARLIK BAZLI EŞİTSİZLİK KISITLARI: min_w <= w_i <= max_w
        symbol_to_idx = {sym: idx for idx, sym in enumerate(symbols)}
        for idx, sym in enumerate(symbols):
            box_row = np.zeros(n_vars, dtype=np.float64)
            box_row[idx] = 1.0
            min_w, max_w = config.asset_bounds.get(sym, (config.min_asset_weight, config.max_asset_weight))
            
            A_rows.append(box_row)
            l_rows.append(float(min_w))
            u_rows.append(float(max_w))
            constraint_names.append(f"Ineq_Box_Asset_{sym}")
            ineq_mask.append(True)

        # C) GRUP MARUZİYET KISITLARI: min_group <= Sum(w_i) <= max_group
        for grp in config.group_constraints:
            group_row = np.zeros(n_vars, dtype=np.float64)
            valid_count = 0
            for sym in grp.symbols:
                if sym in symbol_to_idx:
                    group_row[symbol_to_idx[sym]] = 1.0
                    valid_count += 1
            
            if valid_count > 0:
                A_rows.append(group_row)
                l_rows.append(float(grp.min_weight))
                u_rows.append(float(grp.max_weight))
                constraint_names.append(f"Ineq_Group_{grp.group_name}")
                ineq_mask.append(True)

        # D) ADIM 2.5.1: TURN-OVER VE POZİSYON DEĞİŞİM LINEERLEŞTİRME KISITLARI (u_i >= |w_{t,i} - w_{t-1,i}|)
        if has_turnover and config.w_prev is not None:
            w_prev_dict = config.w_prev
            
            for idx, sym in enumerate(symbols):
                w_prev_val = float(w_prev_dict.get(sym, 0.0))
                
                # 1) w_i - u_i <= w_{t-1, i}  (u_i >= w_i - w_{t-1, i})
                turnover_row_pos = np.zeros(n_vars, dtype=np.float64)
                turnover_row_pos[idx] = 1.0               # w_i
                turnover_row_pos[n_assets + idx] = -1.0    # -u_i
                A_rows.append(turnover_row_pos)
                l_rows.append(-np.inf)
                u_rows.append(w_prev_val)
                constraint_names.append(f"Turnover_Lin_Pos_{sym}")
                ineq_mask.append(True)

                # 2) -w_i - u_i <= -w_{t-1, i}  (u_i >= -(w_i - w_{t-1, i}))
                turnover_row_neg = np.zeros(n_vars, dtype=np.float64)
                turnover_row_neg[idx] = -1.0              # -w_i
                turnover_row_neg[n_assets + idx] = -1.0   # -u_i
                A_rows.append(turnover_row_neg)
                l_rows.append(-np.inf)
                u_rows.append(-w_prev_val)
                constraint_names.append(f"Turnover_Lin_Neg_{sym}")
                ineq_mask.append(True)

                # 3) 0 <= u_i <= max_asset_turnover[sym]  (Kutusal pozisyon değişim sınırı)
                u_box_row = np.zeros(n_vars, dtype=np.float64)
                u_box_row[n_assets + idx] = 1.0
                max_turn_i = float(config.max_asset_turnover.get(sym, np.inf))
                A_rows.append(u_box_row)
                l_rows.append(0.0)
                u_rows.append(max_turn_i)
                constraint_names.append(f"Turnover_Box_{sym}")
                ineq_mask.append(True)

            # 4) Toplam Devir Hızı Sınırı: Sum(u_i) <= max_total_turnover
            if config.max_total_turnover is not None:
                sum_u_row = np.zeros(n_vars, dtype=np.float64)
                sum_u_row[n_assets:] = 1.0
                A_rows.append(sum_u_row)
                l_rows.append(0.0)
                u_rows.append(float(config.max_total_turnover))
                constraint_names.append("Turnover_Sum_Total_Limit")
                ineq_mask.append(True)

        # 3. Matris ve Vektörlerin Birleştirilmesi
        A_arr = np.vstack(A_rows)
        l_arr = np.array(l_rows, dtype=np.float64)
        u_arr = np.array(u_rows, dtype=np.float64)

        return QPProblemInput(
            P=P_full,
            q=q_full,
            A=A_arr,
            l=l_arr,
            u=u_arr,
            symbols=list(symbols),
            constraint_names=constraint_names,
            inequality_mask=np.array(ineq_mask, dtype=bool),
            is_turnover_expanded=has_turnover
        )

