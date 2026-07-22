from __future__ import annotations
"""
AetherForecaster-X v35.2 Multi-Asset Master
Katman 1: Point-in-Time (PIT) Data Lake & Hybrid Risk Engine

Modül: PIT Veri Şeması ve Zamansal Kilitlenme Adaptörü
Adım 1.1.1: Multi-Asset PITDataSnapshot ve PITDataAdapter genişletmesi.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd


class AssetType(Enum):
    """
    Sistem tarafından desteklenen 4 ana hibrit varlık sınıfı.
    """
    EQUITY = "EQUITY"             # Hisse Senetleri (BIST 15-dk barlar, T+2 Takas, 17:52 Kapanış Seansı)
    ETF = "ETF"                   # Borsa Yatırım Fonları (BYF 15-dk barlar, Gold/Foreign/Index, T+2 Takas)
    TEFAS_FREE = "TEFAS_FREE"     # TEFAS Serbest Fonlar (Günlük NAV, TLY vb., T+1/T+2 Valörlü)
    TEFAS_LIQUID = "TEFAS_LIQUID" # TEFAS Likit / Para Piyasa Fonları (Günlük NAV, T+0 Valörlü)

    @property
    def is_intraday(self) -> bool:
        """15 dakikalık yüksek frekanslı bar getirisini destekleyip desteklemediği."""
        return self in (AssetType.EQUITY, AssetType.ETF)

    @property
    def is_daily_nav(self) -> bool:
        """Günlük resmi NAV (Net Asset Value) fiyatlamasına tabi olup olmadığı."""
        return self in (AssetType.TEFAS_FREE, AssetType.TEFAS_LIQUID)


@dataclass(frozen=True)
class DataPoint:
    """
    Tekil bir fiyat/hacim gözlemini temsil eden mühürlenmiş (immutable) veri yapısı.
    """
    symbol: str
    asset_type: AssetType
    timestamp: datetime
    price: float
    volume: float = 0.0
    open_price: Optional[float] = None
    high_price: Optional[float] = None
    low_price: Optional[float] = None
    auction_volume: Optional[float] = None  # Kapanış seansı hacmi (MA20 hesabı için)
    settlement_days: int = 0               # Takas valör gün sayısı (T+0, T+1, T+2)

    def __post_init__(self):
        # Katı veri doğrulama kuralları (Sanity Checks)
        if self.price <= 0:
            raise ValueError(f"Geçersiz Fiyat! {self.symbol} için fiyat <= 0 olamaz: {self.price}")
        if self.volume < 0:
            raise ValueError(f"Geçersiz Hacim! {self.symbol} için hacim < 0 olamaz: {self.volume}")
        if self.auction_volume is not None and self.auction_volume < 0:
            raise ValueError(f"Geçersiz Kapanış Hacmi! {self.symbol}: {self.auction_volume}")
        if self.settlement_days < 0:
            raise ValueError(f"Geçersiz Valör Günü! {self.symbol}: {self.settlement_days}")


@dataclass
class PITDataSnapshot:
    """
    Cuma 17:30 PIT anındaki tüm varlık sınıflarına ait geçmiş verileri barındıran konteyner.
    """
    snapshot_time: datetime
    tefas_cutoff_date: datetime
    equities: Dict[str, List[DataPoint]] = field(default_factory=dict)
    etfs: Dict[str, List[DataPoint]] = field(default_factory=dict)
    tefas_free_funds: Dict[str, List[DataPoint]] = field(default_factory=dict)
    tefas_liquid_funds: Dict[str, List[DataPoint]] = field(default_factory=dict)

    def get_all_symbols(self) -> Set[str]:
        """Snapshot içerisindeki tüm varlık sembollerinin birleşimi."""
        symbols = set(self.equities.keys())
        symbols.update(self.etfs.keys())
        symbols.update(self.tefas_free_funds.keys())
        symbols.update(self.tefas_liquid_funds.keys())
        return symbols

    def get_symbol_asset_type(self, symbol: str) -> AssetType:
        """Sembolün ait olduğu varlık sınıfını döndürür."""
        if symbol in self.equities:
            return AssetType.EQUITY
        elif symbol in self.etfs:
            return AssetType.ETF
        elif symbol in self.tefas_free_funds:
            return AssetType.TEFAS_FREE
        elif symbol in self.tefas_liquid_funds:
            return AssetType.TEFAS_LIQUID
        else:
            raise KeyError(f"Sembol snapshot içinde bulunamadı: {symbol}")

    def get_data_by_asset_type(self, asset_type: AssetType) -> Dict[str, List[DataPoint]]:
        """Varlık tipine göre ilgili veri sözlüğünü döndürür."""
        if asset_type == AssetType.EQUITY:
            return self.equities
        elif asset_type == AssetType.ETF:
            return self.etfs
        elif asset_type == AssetType.TEFAS_FREE:
            return self.tefas_free_funds
        elif asset_type == AssetType.TEFAS_LIQUID:
            return self.tefas_liquid_funds
        else:
            raise ValueError(f"Bilinmeyen varlık tipi: {asset_type}")

    def total_series_count(self) -> int:
        """Toplam aktif varlık serisi sayısı."""
        return len(self.get_all_symbols())

    def to_dataframe(self, asset_type: Optional[AssetType] = None) -> pd.DataFrame:
        """
        Snapshot verilerini Pandas DataFrame formatına dönüştürür.
        """
        records = []
        target_dicts = (
            [self.get_data_by_asset_type(asset_type)]
            if asset_type is not None
            else [self.equities, self.etfs, self.tefas_free_funds, self.tefas_liquid_funds]
        )

        for asset_dict in target_dicts:
            for symbol, points in asset_dict.items():
                for pt in points:
                    records.append({
                        "symbol": pt.symbol,
                        "asset_type": pt.asset_type.value,
                        "timestamp": pt.timestamp,
                        "price": pt.price,
                        "volume": pt.volume,
                        "open_price": pt.open_price,
                        "high_price": pt.high_price,
                        "low_price": pt.low_price,
                        "auction_volume": pt.auction_volume,
                        "settlement_days": pt.settlement_days,
                    })

        if not records:
            return pd.DataFrame(columns=[
                "symbol", "asset_type", "timestamp", "price", "volume",
                "open_price", "high_price", "low_price", "auction_volume", "settlement_days"
            ])

        df = pd.DataFrame(records)
        df.sort_values(by=["symbol", "timestamp"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df


class PITDataAdapter:
    """
    Farklı kaynaklardan (BIST 15dk Bar, TEFAS Günlük NAV) gelen verileri işleyip
    zamansal kilitlenme (Point-in-Time) kurallarına göre doğrulayan adaptör sınıfı.
    """

    def __init__(self, snapshot_time: datetime, tefas_cutoff_date: datetime):
        """
        :param snapshot_time: Hisseler ve BYF'ler için katı Cuma 17:30 PIT zamanı.
        :param tefas_cutoff_date: TEFAS fonları için katı Perşembe 23:59:59 (son resmi NAV) tarihi.
        """
        from aether.data.temporal_lock import TemporalLockEngine, TemporalViolationReport
        self.snapshot_time = snapshot_time
        self.tefas_cutoff_date = tefas_cutoff_date
        self.lock_engine = TemporalLockEngine(snapshot_time=snapshot_time, tefas_cutoff_date=tefas_cutoff_date)

    def build_snapshot_with_audit(self, raw_data_df: pd.DataFrame) -> Tuple[PITDataSnapshot, TemporalViolationReport]:
        """
        Ham veriyi tarar, zamansal kilitlenme filtresini uygular, denetim raporunu üretir
        ve PITDataSnapshot konteynerini döndürür.
        """
        clean_df, report = self.lock_engine.filter_dataframe(raw_data_df)
        snapshot = self.build_snapshot(clean_df)
        self.lock_engine.validate_snapshot(snapshot)
        return snapshot, report

    def build_snapshot(self, raw_data_df: pd.DataFrame) -> PITDataSnapshot:
        """
        Ham pandas DataFrame verisini alır, zamansal kilitlenme filtresini uygular
        ve mühürlenmiş PITDataSnapshot objesini oluşturur.
        """
        required_cols = {"symbol", "asset_type", "timestamp", "price"}
        missing_cols = required_cols - set(raw_data_df.columns)
        if missing_cols:
            raise ValueError(f"Ham veride eksik sütunlar var: {missing_cols}")

        snapshot = PITDataSnapshot(
            snapshot_time=self.snapshot_time,
            tefas_cutoff_date=self.tefas_cutoff_date
        )

        for _, row in raw_data_df.iterrows():
            try:
                asset_type = AssetType(row["asset_type"])
            except ValueError:
                raise ValueError(f"Bilinmeyen varlık tipi stringi: {row['asset_type']}")

            ts = row["timestamp"]
            if isinstance(ts, str):
                ts = pd.to_datetime(ts).to_pydatetime()

            # ZAMANSAL KİLİTLENME VE GELECEK SIZINTISI KONTROLÜ (Adım 1.1.2)
            if asset_type.is_intraday:
                if not self.lock_engine.is_intraday_valid(ts):
                    continue
            elif asset_type.is_daily_nav:
                if not self.lock_engine.is_tefas_valid(ts):
                    continue

            dp = DataPoint(
                symbol=str(row["symbol"]),
                asset_type=asset_type,
                timestamp=ts,
                price=float(row["price"]),
                volume=float(row.get("volume", 0.0)),
                open_price=float(row["open_price"]) if pd.notnull(row.get("open_price")) else None,
                high_price=float(row["high_price"]) if pd.notnull(row.get("high_price")) else None,
                low_price=float(row["low_price"]) if pd.notnull(row.get("low_price")) else None,
                auction_volume=float(row["auction_volume"]) if pd.notnull(row.get("auction_volume")) else None,
                settlement_days=int(row.get("settlement_days", 2 if asset_type == AssetType.EQUITY else (0 if asset_type == AssetType.TEFAS_LIQUID else 1))),
            )

            # Doğru varlık sınıfı sözlüğüne ekleme
            target_dict = snapshot.get_data_by_asset_type(asset_type)
            if dp.symbol not in target_dict:
                target_dict[dp.symbol] = []
            target_dict[dp.symbol].append(dp)

        # Her sembol için verileri zamana göre sırala
        for asset_dict in [snapshot.equities, snapshot.etfs, snapshot.tefas_free_funds, snapshot.tefas_liquid_funds]:
            for symbol in asset_dict:
                asset_dict[symbol].sort(key=lambda x: x.timestamp)

        self.lock_engine.validate_snapshot(snapshot)
        return snapshot
