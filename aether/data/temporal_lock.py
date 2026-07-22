from __future__ import annotations
"""
AetherForecaster-X v35.2 Multi-Asset Master
Katman 1: Point-in-Time (PIT) Data Lake & Hybrid Risk Engine

Modül: Temporal Locking & Leakage Prevention Engine (Zamansal Kilitlenme Motoru)
Adım 1.1.2: Hisseler/BYF'ler için timestamp <= Cuma 17:30:00.000 kilitlenmesi;
            TEFAS fonları için NAV_date <= Perşembe 23:59:59.000 kilitlenmesi.
"""

from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Dict, List, Tuple
import pandas as pd
from aether.data.pit_snapshot import AssetType, DataPoint, PITDataSnapshot


@dataclass
class TemporalViolationReport:
    """
    Zamansal kilitlenme filtresine takılan (sızıntı yapan) kayıtların denetim raporu.
    """
    total_records_checked: int = 0
    total_records_filtered: int = 0
    intraday_violations: int = 0
    tefas_violations: int = 0
    violating_details: List[Dict[str, str]] = field(default_factory=list)

    @property
    def has_violations(self) -> bool:
        return self.total_records_filtered > 0


class TemporalLockEngine:
    """
    Gelecek sızıntısını (Look-Ahead Bias) imkansız kılan katı zamansal kilitlenme motoru.
    """

    def __init__(self, snapshot_time: datetime, tefas_cutoff_date: datetime):
        """
        :param snapshot_time: Hisseler ve BYF'ler için Cuma 17:30:00.000000 katı kilitlenme anı.
        :param tefas_cutoff_date: TEFAS fonları için Perşembe 23:59:59.999999 katı kilitlenme anı.
        """
        self.snapshot_time = snapshot_time
        self.tefas_cutoff_date = tefas_cutoff_date

    def is_intraday_valid(self, timestamp: datetime) -> bool:
        """Hisse/BYF 15dk barının Cuma 17:30:00 sınırını ihlal edip etmediği."""
        return timestamp <= self.snapshot_time

    def is_tefas_valid(self, nav_date: datetime) -> bool:
        """TEFAS fon NAV tarihinin Perşembe 23:59:59 sınırını ihlal edip etmediği."""
        return nav_date <= self.tefas_cutoff_date

    def filter_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, TemporalViolationReport]:
        """
        Ham veritabanı/Pandas DataFrame'ini tarar, zamansal sınırları geçen tüm satırları ayıklar
        ve detaylı denetim raporu ile birlikte temiz DataFrame döndürür.
        """
        report = TemporalViolationReport(total_records_checked=len(df))
        valid_indices = []

        for idx, row in df.iterrows():
            asset_type = AssetType(row["asset_type"])
            ts = row["timestamp"]
            if isinstance(ts, str):
                ts = pd.to_datetime(ts).to_pydatetime()

            if asset_type.is_intraday:
                if not self.is_intraday_valid(ts):
                    report.total_records_filtered += 1
                    report.intraday_violations += 1
                    report.violating_details.append({
                        "symbol": str(row["symbol"]),
                        "asset_type": asset_type.value,
                        "timestamp": str(ts),
                        "limit": str(self.snapshot_time),
                        "reason": f"Hisse/BYF bar zamanı ({ts}) Cuma 17:30 sınırını geçti!"
                    })
                else:
                    valid_indices.append(idx)
            elif asset_type.is_daily_nav:
                if not self.is_tefas_valid(ts):
                    report.total_records_filtered += 1
                    report.tefas_violations += 1
                    report.violating_details.append({
                        "symbol": str(row["symbol"]),
                        "asset_type": asset_type.value,
                        "timestamp": str(ts),
                        "limit": str(self.tefas_cutoff_date),
                        "reason": f"TEFAS NAV tarihi ({ts}) Perşembe 23:59:59 sınırını geçti!"
                    })
                else:
                    valid_indices.append(idx)

        clean_df = df.loc[valid_indices].copy()
        clean_df.reset_index(drop=True, inplace=True)
        return clean_df, report

    def validate_snapshot(self, snapshot: PITDataSnapshot) -> bool:
        """
        Oluşturulmuş bir PITDataSnapshot konteynerinde zamansal ihlal olup olmadığını doğrular.
        Herhangi bir ihlal varsa ValueError fırlatır.
        """
        # Hisseler & BYF'ler kontrolü
        for asset_dict in [snapshot.equities, snapshot.etfs]:
            for symbol, points in asset_dict.items():
                for pt in points:
                    if not self.is_intraday_valid(pt.timestamp):
                        raise ValueError(
                            f"PIT Snapshot İhlali! {symbol} ({pt.asset_type.value}) için "
                            f"zaman {pt.timestamp} > Cuma 17:30 sınırı ({self.snapshot_time})!"
                        )

        # TEFAS Fonları kontrolü
        for fund_dict in [snapshot.tefas_free_funds, snapshot.tefas_liquid_funds]:
            for symbol, points in fund_dict.items():
                for pt in points:
                    if not self.is_tefas_valid(pt.timestamp):
                        raise ValueError(
                            f"PIT Snapshot İhlali! TEFAS Fonu {symbol} ({pt.asset_type.value}) için "
                            f"zaman {pt.timestamp} > Perşembe 23:59:59 sınırı ({self.tefas_cutoff_date})!"
                        )

        return True
