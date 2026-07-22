from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List

# =====================================================================
# MODÜL 1: FİYAT / KAZANÇ ORANI (PE RATIO) ŞEMALARI
# =====================================================================

@dataclass
class PEInputData:
    ticker: str
    timestamp: str
    market_cap: float
    shares_outstanding: float
    current_price: float
    ttm_net_income_gaap: float
    non_recurring_income: float = 0.0
    non_recurring_expenses: float = 0.0
    fwd_earnings_consensus: float = 0.0
    sector_code: str = "GENERAL"
    sector_median_pe: float = 15.0
    risk_free_rate: float = 0.05
    annual_growth_rate_pct: float = 10.0
    net_debt: float = 0.0
    ebitda: float = 1.0
    roic: float = 0.10
    wacc: float = 0.08
    historical_5y_pe_mean: Optional[float] = None
    historical_5y_pe_std: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PEInputData":
        return cls(
            ticker=data["ticker"],
            timestamp=data["timestamp"],
            market_cap=float(data["market_cap"]),
            shares_outstanding=float(data["shares_outstanding"]),
            current_price=float(data["current_price"]),
            ttm_net_income_gaap=float(data["ttm_net_income_gaap"]),
            non_recurring_income=float(data.get("non_recurring_income", 0.0)),
            non_recurring_expenses=float(data.get("non_recurring_expenses", 0.0)),
            fwd_earnings_consensus=float(data.get("fwd_earnings_consensus", 0.0)),
            sector_code=data.get("sector_code", "GENERAL"),
            sector_median_pe=float(data.get("sector_median_pe", 15.0)),
            risk_free_rate=float(data.get("risk_free_rate", 0.35 if data.get("sector_code") == "STEEL" else 0.05)),
            annual_growth_rate_pct=float(data.get("annual_growth_rate_pct", 10.0)),
            net_debt=float(data.get("net_debt", 0.0)),
            ebitda=float(data.get("ebitda", 1.0)),
            roic=float(data.get("roic", 0.10)),
            wacc=float(data.get("wacc", 0.08)),
            historical_5y_pe_mean=float(data["historical_5y_pe_mean"]) if data.get("historical_5y_pe_mean") is not None else None,
            historical_5y_pe_std=float(data["historical_5y_pe_std"]) if data.get("historical_5y_pe_std") is not None else None,
        )

@dataclass
class PEFlags:
    is_cyclical_trap_risk: bool = False
    one_off_income_detected: bool = False
    is_distressed_pe: bool = False
    is_debt_trap_risk: bool = False
    is_bubble_pricing: bool = False
    is_negative_earnings: bool = False

    def to_dict(self) -> Dict[str, bool]:
        return {
            "is_cyclical_trap_risk": self.is_cyclical_trap_risk,
            "one_off_income_detected": self.one_off_income_detected,
            "is_distressed_pe": self.is_distressed_pe,
            "is_debt_trap_risk": self.is_debt_trap_risk,
            "is_bubble_pricing": self.is_bubble_pricing,
            "is_negative_earnings": self.is_negative_earnings,
        }

@dataclass
class PEOutputData:
    ticker: str
    raw_pe: Optional[float]
    adjusted_pe: Optional[float]
    earnings_yield: float
    peg_ratio: Optional[float]
    flags: Dict[str, bool]
    signal_score: float
    confidence_interval: float
    primary_recommendation_contribution: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "raw_pe": round(self.raw_pe, 2) if self.raw_pe is not None else None,
            "adjusted_pe": round(self.adjusted_pe, 2) if self.adjusted_pe is not None else None,
            "earnings_yield": round(self.earnings_yield, 4),
            "peg_ratio": round(self.peg_ratio, 2) if self.peg_ratio is not None else None,
            "flags": self.flags,
            "signal_score": round(self.signal_score, 2),
            "confidence_interval": round(self.confidence_interval, 2),
            "primary_recommendation_contribution": self.primary_recommendation_contribution
        }


# =====================================================================
# MODÜL 2: PD/DD ORANI (PB RATIO) ŞEMALARI
# =====================================================================

@dataclass
class PBInputData:
    ticker: str
    timestamp: str
    market_cap: float
    total_assets: float
    total_liabilities: float
    goodwill: float = 0.0
    intangible_assets: float = 0.0
    deferred_tax_assets: float = 0.0
    return_on_equity: float = 0.15
    cost_of_equity: float = 0.12
    sector_median_pb: float = 2.0
    sustainable_growth_rate: float = 0.06
    revalued_asset_adjustment: float = 0.0
    net_debt: float = 0.0
    is_asset_light: bool = False
    corporate_governance_score: float = 0.80

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PBInputData":
        return cls(
            ticker=data["ticker"],
            timestamp=data["timestamp"],
            market_cap=float(data["market_cap"]),
            total_assets=float(data["total_assets"]),
            total_liabilities=float(data["total_liabilities"]),
            goodwill=float(data.get("goodwill", 0.0)),
            intangible_assets=float(data.get("intangible_assets", 0.0)),
            deferred_tax_assets=float(data.get("deferred_tax_assets", 0.0)),
            return_on_equity=float(data.get("return_on_equity", 0.15)),
            cost_of_equity=float(data.get("cost_of_equity", 0.12)),
            sector_median_pb=float(data.get("sector_median_pb", 2.0)),
            sustainable_growth_rate=float(data.get("sustainable_growth_rate", 0.06)),
            revalued_asset_adjustment=float(data.get("revalued_asset_adjustment", 0.0)),
            net_debt=float(data.get("net_debt", 0.0)),
            is_asset_light=bool(data.get("is_asset_light", False)),
            corporate_governance_score=float(data.get("corporate_governance_score", 0.80)),
        )

@dataclass
class PBFlags:
    is_tangible_discount: bool = False
    value_trap_risk: bool = False
    unrealized_asset_value_potential: bool = False
    is_negative_equity: bool = False
    is_goodwill_inflated_trap: bool = False
    is_leverage_risk: bool = False

    def to_dict(self) -> Dict[str, bool]:
        return {
            "is_tangible_discount": self.is_tangible_discount,
            "value_trap_risk": self.value_trap_risk,
            "unrealized_asset_value_potential": self.unrealized_asset_value_potential,
            "is_negative_equity": self.is_negative_equity,
            "is_goodwill_inflated_trap": self.is_goodwill_inflated_trap,
            "is_leverage_risk": self.is_leverage_risk
        }

@dataclass
class PBOutputData:
    ticker: str
    raw_pb: Optional[float]
    tangible_pb: Optional[float]
    justified_pb: Optional[float]
    discount_to_justified: float
    flags: Dict[str, bool]
    safety_margin_score: float
    primary_recommendation_contribution: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "raw_pb": round(self.raw_pb, 2) if self.raw_pb is not None else None,
            "tangible_pb": round(self.tangible_pb, 2) if self.tangible_pb is not None else None,
            "justified_pb": round(self.justified_pb, 2) if self.justified_pb is not None else None,
            "discount_to_justified": round(self.discount_to_justified, 3),
            "flags": self.flags,
            "safety_margin_score": round(self.safety_margin_score, 2),
            "primary_recommendation_contribution": self.primary_recommendation_contribution
        }


# =====================================================================
# MODÜL 3: FAVÖK / EBITDA İŞLEME ŞEMALARI
# =====================================================================

@dataclass
class EBITDAInputData:
    ticker: str
    timestamp: str
    revenue: float
    ebit: float
    depreciation_amortization: float
    ifrs16_lease_payments: float = 0.0
    stock_based_compensation: float = 0.0
    capitalized_capex: float = 0.0
    operating_cash_flow: float = 0.0
    capex: float = 0.0
    net_debt: float = 0.0
    net_income_gaap: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EBITDAInputData":
        return cls(
            ticker=data["ticker"],
            timestamp=data["timestamp"],
            revenue=float(data["revenue"]),
            ebit=float(data["ebit"]),
            depreciation_amortization=float(data["depreciation_amortization"]),
            ifrs16_lease_payments=float(data.get("ifrs16_lease_payments", 0.0)),
            stock_based_compensation=float(data.get("stock_based_compensation", 0.0)),
            capitalized_capex=float(data.get("capitalized_capex", 0.0)),
            operating_cash_flow=float(data.get("operating_cash_flow", 0.0)),
            capex=float(data.get("capex", 0.0)),
            net_debt=float(data.get("net_debt", 0.0)),
            net_income_gaap=float(data.get("net_income_gaap", 0.0)),
        )

@dataclass
class EBITDAFlags:
    ifrs16_distortion_high: bool = False
    sbc_dilution_risk: bool = False
    capitalized_capex_warning: bool = False
    cash_flow_decoupling_risk: bool = False
    turnaround_candidate: bool = False
    high_leverage_warning: bool = False

    def to_dict(self) -> Dict[str, bool]:
        return {
            "ifrs16_distortion_high": self.ifrs16_distortion_high,
            "sbc_dilution_risk": self.sbc_dilution_risk,
            "capitalized_capex_warning": self.capitalized_capex_warning,
            "cash_flow_decoupling_risk": self.cash_flow_decoupling_risk,
            "turnaround_candidate": self.turnaround_candidate,
            "high_leverage_warning": self.high_leverage_warning,
        }

@dataclass
class EBITDAOutputData:
    ticker: str
    raw_ebitda: float
    adjusted_ebitda: float
    ebitda_margin: float
    cash_conversion_rate: float
    net_debt_to_ebitda: float
    flags: Dict[str, bool]
    operational_quality_score: float
    primary_recommendation_contribution: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "raw_ebitda": round(self.raw_ebitda, 2),
            "adjusted_ebitda": round(self.adjusted_ebitda, 2),
            "ebitda_margin": round(self.ebitda_margin, 3),
            "cash_conversion_rate": round(self.cash_conversion_rate, 2),
            "net_debt_to_ebitda": round(self.net_debt_to_ebitda, 2),
            "flags": self.flags,
            "operational_quality_score": round(self.operational_quality_score, 2),
            "primary_recommendation_contribution": self.primary_recommendation_contribution
        }


# =====================================================================
# MODÜL 4: FİRMA DEĞERİ / FAVÖK (EV / EBITDA) ŞEMALARI
# =====================================================================

@dataclass
class EVInputData:
    ticker: str
    timestamp: str
    market_cap: float
    gross_debt: float
    lease_liabilities: float = 0.0
    total_cash: float = 0.0
    restricted_cash: float = 0.0
    minority_interest: float = 0.0
    associates_value: float = 0.0
    adjusted_ebitda: float = 1.0
    capex: float = 0.0
    sector_median_ev_ebitda: float = 8.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EVInputData":
        return cls(
            ticker=data["ticker"],
            timestamp=data["timestamp"],
            market_cap=float(data["market_cap"]),
            gross_debt=float(data["gross_debt"]),
            lease_liabilities=float(data.get("lease_liabilities", 0.0)),
            total_cash=float(data.get("total_cash", 0.0)),
            restricted_cash=float(data.get("restricted_cash", 0.0)),
            minority_interest=float(data.get("minority_interest", 0.0)),
            associates_value=float(data.get("associates_value", 0.0)),
            adjusted_ebitda=float(data.get("adjusted_ebitda", 1.0)),
            capex=float(data.get("capex", 0.0)),
            sector_median_ev_ebitda=float(data.get("sector_median_ev_ebitda", 8.0)),
        )

@dataclass
class EVFlags:
    is_net_cash_company: bool = False
    capex_trap_risk: bool = False
    restricted_cash_adjusted: bool = False
    ma_target_potential: bool = False
    refinancing_risk_flag: bool = False
    is_negative_ebitda: bool = False

    def to_dict(self) -> Dict[str, bool]:
        return {
            "is_net_cash_company": self.is_net_cash_company,
            "capex_trap_risk": self.capex_trap_risk,
            "restricted_cash_adjusted": self.restricted_cash_adjusted,
            "ma_target_potential": self.ma_target_potential,
        }

@dataclass
class EVOutputData:
    ticker: str
    enterprise_value: Optional[float]
    ev_to_adjusted_ebitda: Optional[float]
    ev_to_ebitda_minus_capex: Optional[float]
    leverage_to_ev_ratio: float
    flags: Dict[str, bool]
    valuation_attractiveness_score: float
    primary_recommendation_contribution: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "enterprise_value": round(self.enterprise_value, 2) if self.enterprise_value is not None else None,
            "ev_to_adjusted_ebitda": round(self.ev_to_adjusted_ebitda, 2) if self.ev_to_adjusted_ebitda is not None else None,
            "ev_to_ebitda_minus_capex": round(self.ev_to_ebitda_minus_capex, 2) if self.ev_to_ebitda_minus_capex is not None else None,
            "leverage_to_ev_ratio": round(self.leverage_to_ev_ratio, 2),
            "flags": self.flags,
            "valuation_attractiveness_score": round(self.valuation_attractiveness_score, 2),
            "primary_recommendation_contribution": self.primary_recommendation_contribution
        }


# =====================================================================
# MODÜL 5: ÖZVARLIK KÂRLILIĞI (ROE & DUPONT) ŞEMALARI
# =====================================================================

@dataclass
class ROEInputData:
    ticker: str
    timestamp: str
    net_income_ttm: float
    equity_t0: float
    equity_t4: float
    revenue_ttm: float
    total_assets: float
    ebit_ttm: float
    one_off_income: float = 0.0
    inflation_rate: float = 0.30
    risk_free_rate: float = 0.25
    cost_of_equity: float = 0.35
    payout_ratio: float = 0.50

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ROEInputData":
        return cls(
            ticker=data["ticker"],
            timestamp=data["timestamp"],
            net_income_ttm=float(data["net_income_ttm"]),
            equity_t0=float(data["equity_t0"]),
            equity_t4=float(data["equity_t4"]),
            revenue_ttm=float(data["revenue_ttm"]),
            total_assets=float(data["total_assets"]),
            ebit_ttm=float(data["ebit_ttm"]),
            one_off_income=float(data.get("one_off_income", 0.0)),
            inflation_rate=float(data.get("inflation_rate", 0.30)),
            risk_free_rate=float(data.get("risk_free_rate", 0.25)),
            cost_of_equity=float(data.get("cost_of_equity", 0.35)),
            payout_ratio=float(data.get("payout_ratio", 0.50)),
        )

@dataclass
class DuPontBreakdown:
    tax_burden: float
    interest_burden: float
    ebit_margin: float
    asset_turnover: float
    financial_leverage: float
    primary_driver: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tax_burden": round(self.tax_burden, 2),
            "interest_burden": round(self.interest_burden, 2),
            "ebit_margin": round(self.ebit_margin, 3),
            "asset_turnover": round(self.asset_turnover, 2),
            "financial_leverage": round(self.financial_leverage, 2),
            "primary_driver": self.primary_driver
        }

@dataclass
class ROEFlags:
    double_negative_trap: bool = False
    leverage_driven_risk: bool = False
    capital_destruction_risk: bool = False

    def to_dict(self) -> Dict[str, bool]:
        return {
            "double_negative_trap": self.double_negative_trap,
            "leverage_driven_risk": self.leverage_driven_risk,
            "capital_destruction_risk": self.capital_destruction_risk
        }

@dataclass
class ROEOutputData:
    ticker: str
    raw_roe: Optional[float]
    core_roe: Optional[float]
    real_roe_spread: Optional[float]
    dupont_analysis: Dict[str, Any]
    sustainable_growth_rate: Optional[float]
    justified_pb_ratio: Optional[float]
    flags: Dict[str, bool]
    capital_quality_score: float
    primary_recommendation_contribution: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "raw_roe": round(self.raw_roe, 2) if self.raw_roe is not None else None,
            "core_roe": round(self.core_roe, 2) if self.core_roe is not None else None,
            "real_roe_spread": round(self.real_roe_spread, 2) if self.real_roe_spread is not None else None,
            "dupont_analysis": self.dupont_analysis,
            "sustainable_growth_rate": round(self.sustainable_growth_rate, 2) if self.sustainable_growth_rate is not None else None,
            "justified_pb_ratio": round(self.justified_pb_ratio, 2) if self.justified_pb_ratio is not None else None,
            "flags": self.flags,
            "capital_quality_score": round(self.capital_quality_score, 2),
            "primary_recommendation_contribution": self.primary_recommendation_contribution
        }


# =====================================================================
# MODÜL 6: CARİ ORAN (CURRENT RATIO & SOLVENCY) ŞEMALARI
# =====================================================================

@dataclass
class CurrentRatioInputData:
    ticker: str
    timestamp: str
    sector_code: str
    current_assets: float
    current_liabilities: float
    inventories: float = 0.0
    accounts_receivable: float = 0.0
    related_party_receivables: float = 0.0
    doubtful_receivables: float = 0.0
    cash_and_equivalents: float = 0.0
    operating_cash_flow: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CurrentRatioInputData":
        return cls(
            ticker=data["ticker"],
            timestamp=data["timestamp"],
            sector_code=data.get("sector_code", "GENERAL"),
            current_assets=float(data["current_assets"]),
            current_liabilities=float(data["current_liabilities"]),
            inventories=float(data.get("inventories", 0.0)),
            accounts_receivable=float(data.get("accounts_receivable", 0.0)),
            related_party_receivables=float(data.get("related_party_receivables", 0.0)),
            doubtful_receivables=float(data.get("doubtful_receivables", 0.0)),
            cash_and_equivalents=float(data.get("cash_and_equivalents", 0.0)),
            operating_cash_flow=float(data.get("operating_cash_flow", 0.0)),
        )

@dataclass
class CurrentRatioFlags:
    inventory_heavy_liquidity: bool = False
    related_party_receivable_risk: bool = False
    liquidity_distress_warning: bool = False
    retail_model_exception: bool = False

    def to_dict(self) -> Dict[str, bool]:
        return {
            "inventory_heavy_liquidity": self.inventory_heavy_liquidity,
            "related_party_receivable_risk": self.related_party_receivable_risk,
            "liquidity_distress_warning": self.liquidity_distress_warning,
            "retail_model_exception": self.retail_model_exception
        }

@dataclass
class CurrentRatioOutputData:
    ticker: str
    raw_current_ratio: float
    adjusted_current_ratio: float
    quick_ratio: float
    cash_ratio: float
    net_working_capital: float
    flags: Dict[str, bool]
    solvency_health_score: float
    primary_recommendation_contribution: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "raw_current_ratio": round(self.raw_current_ratio, 3),
            "adjusted_current_ratio": round(self.adjusted_current_ratio, 2),
            "quick_ratio": round(self.quick_ratio, 3),
            "cash_ratio": round(self.cash_ratio, 2),
            "net_working_capital": round(self.net_working_capital, 2),
            "flags": self.flags,
            "solvency_health_score": round(self.solvency_health_score, 2),
            "primary_recommendation_contribution": self.primary_recommendation_contribution
        }


# =====================================================================
# MODÜL 7: GÖRECELİ GÜÇ ENDEKSİ (RSI & TECHNICAL) ŞEMALARI
# =====================================================================

@dataclass
class OHLCVBar:
    timestamp: str
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OHLCVBar":
        return cls(
            timestamp=data["timestamp"],
            open=float(data.get("open", 0.0)),
            high=float(data.get("high", 0.0)),
            low=float(data.get("low", 0.0)),
            close=float(data["close"]),
            volume=float(data.get("volume", 0.0))
        )

@dataclass
class DivergenceResult:
    detected: bool
    type: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detected": self.detected,
            "type": self.type,
            "confidence": round(self.confidence, 2)
        }

@dataclass
class RSIInputData:
    ticker: str
    timeframe: str
    timestamp: str
    period: int
    ohlcv_data: List[OHLCVBar]
    adx_val: float = 20.0
    atr_val: float = 5.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RSIInputData":
        bars = [OHLCVBar.from_dict(b) for b in data["ohlcv_data"]]
        return cls(
            ticker=data["ticker"],
            timeframe=data.get("timeframe", "1D"),
            timestamp=data["timestamp"],
            period=int(data.get("period", 14)),
            ohlcv_data=bars,
            adx_val=float(data.get("adx_val", 20.0)),
            atr_val=float(data.get("atr_val", 5.0))
        )

@dataclass
class RSIFlags:
    is_overbought: bool = False
    is_oversold: bool = False
    momentum_exhaustion_warning: bool = False
    trend_override_active: bool = False

    def to_dict(self) -> Dict[str, bool]:
        return {
            "is_overbought": self.is_overbought,
            "is_oversold": self.is_oversold,
            "momentum_exhaustion_warning": self.momentum_exhaustion_warning,
            "trend_override_active": self.trend_override_active
        }

@dataclass
class RSIOutputData:
    ticker: str
    timeframe: str
    raw_rsi: float
    dynamic_overbought_threshold: float
    dynamic_oversold_threshold: float
    rsi_velocity: float
    divergence: Dict[str, Any]
    flags: Dict[str, bool]
    technical_signal_score: float
    primary_recommendation_contribution: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "timeframe": self.timeframe,
            "raw_rsi": round(self.raw_rsi, 1),
            "dynamic_overbought_threshold": round(self.dynamic_overbought_threshold, 1),
            "dynamic_oversold_threshold": round(self.dynamic_oversold_threshold, 1),
            "rsi_velocity": round(self.rsi_velocity, 1),
            "divergence": self.divergence,
            "flags": self.flags,
            "technical_signal_score": round(self.technical_signal_score, 2),
            "primary_recommendation_contribution": self.primary_recommendation_contribution
        }


# =====================================================================
# MODÜL 8: HAREKETLİ ORTALAMALAR (MA 50/200 & CROSS) ŞEMALARI
# =====================================================================

@dataclass
class CrossStatus:
    event: str
    candles_since_cross: int
    is_confirmed: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": self.event,
            "candles_since_cross": self.candles_since_cross,
            "is_confirmed": self.is_confirmed
        }

@dataclass
class DistanceMetrics:
    distance_to_200ma_pct: float
    distance_z_score: float
    mean_reversion_risk: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "distance_to_200ma_pct": round(self.distance_to_200ma_pct, 3),
            "distance_z_score": round(self.distance_z_score, 2),
            "mean_reversion_risk": self.mean_reversion_risk
        }

@dataclass
class MAInputData:
    ticker: str
    timeframe: str
    timestamp: str
    ma_short_period: int
    ma_long_period: int
    ma_type: str
    ohlcv_data: List[OHLCVBar]
    adx_val: float = 25.0
    atr_val: float = 3.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MAInputData":
        bars = [OHLCVBar.from_dict(b) for b in data["ohlcv_data"]]
        return cls(
            ticker=data["ticker"],
            timeframe=data.get("timeframe", "1D"),
            timestamp=data["timestamp"],
            ma_short_period=int(data.get("ma_short_period", 50)),
            ma_long_period=int(data.get("ma_long_period", 200)),
            ma_type=data.get("ma_type", "EMA"),
            ohlcv_data=bars,
            adx_val=float(data.get("adx_val", 25.0)),
            atr_val=float(data.get("atr_val", 3.0))
        )

@dataclass
class MAFlags:
    is_whipsaw_risk: bool = False
    dynamic_support_bounce: bool = False
    overextended_warning: bool = False

    def to_dict(self) -> Dict[str, bool]:
        return {
            "is_whipsaw_risk": self.is_whipsaw_risk,
            "dynamic_support_bounce": self.dynamic_support_bounce,
            "overextended_warning": self.overextended_warning
        }

@dataclass
class MAOutputData:
    ticker: str
    timeframe: str
    ma_50_value: float
    ma_200_value: float
    ma_spread_pct: float
    ma_200_slope: float
    cross_status: Dict[str, Any]
    distance_metrics: Dict[str, Any]
    flags: Dict[str, bool]
    trend_state_score: float
    primary_recommendation_contribution: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "timeframe": self.timeframe,
            "ma_50_value": round(self.ma_50_value, 1),
            "ma_200_value": round(self.ma_200_value, 1),
            "ma_spread_pct": round(self.ma_spread_pct, 3),
            "ma_200_slope": round(self.ma_200_slope, 3),
            "cross_status": self.cross_status,
            "distance_metrics": self.distance_metrics,
            "flags": self.flags,
            "trend_state_score": round(self.trend_state_score, 2),
            "primary_recommendation_contribution": self.primary_recommendation_contribution
        }


# =====================================================================
# MODÜL 9: MACD & İVMELENME (TECHNICAL PPO MACD) ŞEMALARI
# =====================================================================

@dataclass
class CrossoverEvents:
    bullish_cross_active: bool
    candles_since_cross: int
    cross_quality: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bullish_cross_active": self.bullish_cross_active,
            "candles_since_cross": self.candles_since_cross,
            "cross_quality": self.cross_quality
        }

@dataclass
class MACDInputData:
    ticker: str
    timeframe: str
    timestamp: str
    fast_period: int
    slow_period: int
    signal_period: int
    ohlcv_data: List[OHLCVBar]
    adx_val: float = 25.0
    atr_val: float = 2.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MACDInputData":
        bars = [OHLCVBar.from_dict(b) for b in data["ohlcv_data"]]
        return cls(
            ticker=data["ticker"],
            timeframe=data.get("timeframe", "4h"),
            timestamp=data["timestamp"],
            fast_period=int(data.get("fast_period", 12)),
            slow_period=int(data.get("slow_period", 26)),
            signal_period=int(data.get("signal_period", 9)),
            ohlcv_data=bars,
            adx_val=float(data.get("adx_val", 25.0)),
            atr_val=float(data.get("atr_val", 2.0))
        )

@dataclass
class MACDFlags:
    is_decelerating: bool = False
    whipsaw_risk: bool = False
    overbought_ppo_warning: bool = False

    def to_dict(self) -> Dict[str, bool]:
        return {
            "is_decelerating": self.is_decelerating,
            "whipsaw_risk": self.whipsaw_risk,
            "overbought_ppo_warning": self.overbought_ppo_warning
        }

@dataclass
class MACDOutputData:
    ticker: str
    timeframe: str
    ppo_line: float
    signal_line: float
    histogram_value: float
    delta_histogram: float
    histogram_acceleration: float
    zero_line_status: str
    crossover_events: Dict[str, Any]
    divergence: Dict[str, Any]
    flags: Dict[str, bool]
    momentum_score: float
    primary_recommendation_contribution: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "timeframe": self.timeframe,
            "ppo_line": round(self.ppo_line, 2),
            "signal_line": round(self.signal_line, 2),
            "histogram_value": round(self.histogram_value, 2),
            "delta_histogram": round(self.delta_histogram, 2),
            "histogram_acceleration": round(self.histogram_acceleration, 2),
            "zero_line_status": self.zero_line_status,
            "crossover_events": self.crossover_events,
            "divergence": self.divergence,
            "flags": self.flags,
            "momentum_score": round(self.momentum_score, 2),
            "primary_recommendation_contribution": self.primary_recommendation_contribution
        }


# =====================================================================
# MODÜL 10: BOLLINGER BANTLARI & VOLATİLİTE (BOLLINGER BANDS) ŞEMALARI
# =====================================================================

@dataclass
class SqueezeStatus:
    is_squeezing: bool
    squeeze_duration_candles: int
    breakout_direction: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_squeezing": self.is_squeezing,
            "squeeze_duration_candles": self.squeeze_duration_candles,
            "breakout_direction": self.breakout_direction
        }

@dataclass
class BollingerInputData:
    ticker: str
    timeframe: str
    timestamp: str
    period: int
    k_std_dev: float
    ohlcv_data: List[OHLCVBar]
    adx_val: float = 34.2
    atr_val: float = 1.85

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BollingerInputData":
        bars = [OHLCVBar.from_dict(b) for b in data["ohlcv_data"]]
        return cls(
            ticker=data["ticker"],
            timeframe=data.get("timeframe", "1D"),
            timestamp=data["timestamp"],
            period=int(data.get("period", 20)),
            k_std_dev=float(data.get("k_std_dev", 2.0)),
            ohlcv_data=bars,
            adx_val=float(data.get("adx_val", 34.2)),
            atr_val=float(data.get("atr_val", 1.85))
        )

@dataclass
class BollingerFlags:
    band_walking_active: bool = False
    mean_reversion_risk: bool = False
    head_fake_warning: bool = False
    volume_confirmed: bool = False

    def to_dict(self) -> Dict[str, bool]:
        return {
            "band_walking_active": self.band_walking_active,
            "mean_reversion_risk": self.mean_reversion_risk,
            "head_fake_warning": self.head_fake_warning,
            "volume_confirmed": self.volume_confirmed
        }

@dataclass
class BollingerOutputData:
    ticker: str
    timeframe: str
    upper_band: float
    middle_band: float
    lower_band: float
    bandwidth: float
    percent_b: float
    bw_z_score: float
    squeeze_status: Dict[str, Any]
    flags: Dict[str, bool]
    volatility_breakout_score: float
    primary_recommendation_contribution: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "timeframe": self.timeframe,
            "upper_band": round(self.upper_band, 2),
            "middle_band": round(self.middle_band, 2),
            "lower_band": round(self.lower_band, 2),
            "bandwidth": round(self.bandwidth, 3),
            "percent_b": round(self.percent_b, 3),
            "bw_z_score": round(self.bw_z_score, 2),
            "squeeze_status": self.squeeze_status,
            "flags": self.flags,
            "volatility_breakout_score": round(self.volatility_breakout_score, 2),
            "primary_recommendation_contribution": self.primary_recommendation_contribution
        }


# =====================================================================
# MODÜL 11: HACİM & OBV (VOLUME & ON-BALANCE VOLUME) ŞEMALARI
# =====================================================================

@dataclass
class VolumeInputData:
    ticker: str
    timeframe: str
    timestamp: str
    ohlcv_data: List[OHLCVBar]
    bid_volume: float = 0.0
    ask_volume: float = 0.0
    market_cap_tier: str = "LARGE_CAP"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VolumeInputData":
        bars = [OHLCVBar.from_dict(b) for b in data["ohlcv_data"]]
        return cls(
            ticker=data["ticker"],
            timeframe=data.get("timeframe", "1D"),
            timestamp=data["timestamp"],
            ohlcv_data=bars,
            bid_volume=float(data.get("bid_volume", 0.0)),
            ask_volume=float(data.get("ask_volume", 0.0)),
            market_cap_tier=data.get("market_cap_tier", "LARGE_CAP")
        )

@dataclass
class VolumeFlags:
    is_volume_spike: bool = False
    wash_trading_risk: bool = False
    illiquid_stock_trap: bool = False
    institutional_buying_detected: bool = False

    def to_dict(self) -> Dict[str, bool]:
        return {
            "is_volume_spike": self.is_volume_spike,
            "wash_trading_risk": self.wash_trading_risk,
            "illiquid_stock_trap": self.illiquid_stock_trap,
            "institutional_buying_detected": self.institutional_buying_detected
        }

@dataclass
class VolumeOutputData:
    ticker: str
    timeframe: str
    obv_value: float
    obv_ema_20: float
    rvol_normalized: float
    bid_ask_delta: float
    divergence: Dict[str, Any]
    flags: Dict[str, bool]
    smart_money_flow_score: float
    primary_recommendation_contribution: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "timeframe": self.timeframe,
            "obv_value": round(self.obv_value, 0),
            "obv_ema_20": round(self.obv_ema_20, 0),
            "rvol_normalized": round(self.rvol_normalized, 2),
            "bid_ask_delta": round(self.bid_ask_delta, 0),
            "divergence": self.divergence,
            "flags": self.flags,
            "smart_money_flow_score": round(self.smart_money_flow_score, 2),
            "primary_recommendation_contribution": self.primary_recommendation_contribution
        }


# =====================================================================
# MODÜL 12: FIBONACCI DÜZELTME & KESİŞİM (FIBONACCI & CONFLUENCE) ŞEMALARI
# =====================================================================

@dataclass
class FibAnchors:
    swing_high: float
    swing_low: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "swing_high": round(self.swing_high, 2),
            "swing_low": round(self.swing_low, 2)
        }

@dataclass
class FibLevels:
    level_0236: float
    level_0382: float
    level_0500: float
    golden_pocket_0618_0650: List[float]
    level_0786: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level_0236": round(self.level_0236, 2),
            "level_0382": round(self.level_0382, 2),
            "level_0500": round(self.level_0500, 2),
            "golden_pocket_0618_0650": [round(x, 2) for x in self.golden_pocket_0618_0650],
            "level_0786": round(self.level_0786, 2)
        }

@dataclass
class ConfluenceAnalysis:
    active_zone: str
    confluence_score: int
    matched_elements: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_zone": self.active_zone,
            "confluence_score": self.confluence_score,
            "matched_elements": self.matched_elements
        }


# =====================================================================
# FAZ 22: AÇIKLANABİLİR KARAR DESTEK SİSTEMİ (EXPLAINABLE DSS) ŞEMALARI
# =====================================================================

@dataclass
class AdvisorySummary:
    action_recommendation: str              # ACCUMULATE_BUY, STRONG_BUY, NEUTRAL_HOLD, REDUCE
    confidence_level_pct: float             # Örn: 78.5
    urgency_level: str                      # MEDIUM_TERM_SWING, IMMEDIATE, SCALPING
    recommended_portfolio_weight_pct: float # Örn: 6.5
    suggested_entry_zone: List[float]       # [284.50, 286.80]
    suggested_stop_loss: float              # 272.10
    suggested_take_profit_target: float     # 325.00
    risk_reward_ratio: float                # 2.85

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_recommendation": self.action_recommendation,
            "confidence_level_pct": round(self.confidence_level_pct, 1),
            "urgency_level": self.urgency_level,
            "recommended_portfolio_weight_pct": round(self.recommended_portfolio_weight_pct, 2),
            "suggested_entry_zone": [round(x, 2) for x in self.suggested_entry_zone],
            "suggested_stop_loss": round(self.suggested_stop_loss, 2),
            "suggested_take_profit_target": round(self.suggested_take_profit_target, 2),
            "risk_reward_ratio": round(self.risk_reward_ratio, 2)
        }

@dataclass
class SHAPFactor:
    factor: str
    impact_score: float
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "factor": self.factor,
            "impact_score": round(self.impact_score, 2),
            "description": self.description
        }

@dataclass
class ExplainableAIDrivers:
    top_positive_factors: List[SHAPFactor]
    top_negative_factors: List[SHAPFactor]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "top_positive_factors": [x.to_dict() for x in self.top_positive_factors],
            "top_negative_factors": [x.to_dict() for x in self.top_negative_factors]
        }

@dataclass
class ExecutionAdvisorGuidance:
    liquidity_status: str                   # MODERATE_LIQUIDITY, HIGH_LIQUIDITY, LOW_FLOAT_WARNING
    recommended_order_type: str             # LIMIT_ORDER_TRANCHES, TWAP_ICEBERG, MARKET_ORDER
    estimated_slippage_if_market_order_pct: float
    execution_tip: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "liquidity_status": self.liquidity_status,
            "recommended_order_type": self.recommended_order_type,
            "estimated_slippage_if_market_order_pct": round(self.estimated_slippage_if_market_order_pct, 2),
            "execution_tip": self.execution_tip
        }

@dataclass
class WhatIfPortfolioImpact:
    current_portfolio_var_pct: float
    projected_portfolio_var_pct: float
    sector_exposure_after_trade: Dict[str, str]
    correlation_warning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_portfolio_var_pct": round(self.current_portfolio_var_pct, 2),
            "projected_portfolio_var_pct": round(self.projected_portfolio_var_pct, 2),
            "sector_exposure_after_trade": self.sector_exposure_after_trade,
            "correlation_warning": self.correlation_warning
        }

@dataclass
class UserDecisionCapture:
    status: str                            # AWAITING_HUMAN_APPROVAL, APPROVED, REJECTED
    user_action: Optional[str] = None       # ACCEPT, OVERRULE
    user_override_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "user_action": self.user_action,
            "user_override_reason": self.user_override_reason
        }

@dataclass
class AdvisoryCardOutputData:
    ticker: str
    timestamp: str
    advisory_summary: AdvisorySummary
    explainable_ai_drivers: ExplainableAIDrivers
    execution_advisor_guidance: ExecutionAdvisorGuidance
    what_if_portfolio_impact: WhatIfPortfolioImpact
    user_decision_capture: UserDecisionCapture

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp,
            "advisory_summary": self.advisory_summary.to_dict(),
            "explainable_ai_drivers": self.explainable_ai_drivers.to_dict(),
            "execution_advisor_guidance": self.execution_advisor_guidance.to_dict(),
            "what_if_portfolio_impact": self.what_if_portfolio_impact.to_dict(),
            "user_decision_capture": self.user_decision_capture.to_dict()
        }


# =====================================================================
# MUTLAK MAKRO REJİM KATMANI (ABSOLUTE MACRO OVERLAY) ŞEMALARI
# =====================================================================

@dataclass
class MacroOverlayInputData:
    timestamp: str
    bist100_index_price: float
    bist100_ema_200: float
    volatility_index_vix: float
    volatility_threshold: float = 30.0
    systemic_crash_flag: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MacroOverlayInputData":
        return cls(
            timestamp=data["timestamp"],
            bist100_index_price=float(data["bist100_index_price"]),
            bist100_ema_200=float(data["bist100_ema_200"]),
            volatility_index_vix=float(data.get("volatility_index_vix", 20.0)),
            volatility_threshold=float(data.get("volatility_threshold", 30.0)),
            systemic_crash_flag=bool(data.get("systemic_crash_flag", False))
        )

@dataclass
class MacroOverlayOutputData:
    macro_state: int                      # 1: Normal/Bull, 0: Systemic Crash
    is_cash_protection_active: bool
    macro_regime_label: str               # NORMAL_MARKET_REGIME, FORCE_CASH_SYSTEMIC_CRASH_PROTECTION
    risk_multiplier: float                # 1.0 (Normal) or 0.0 (Crash)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "macro_state": self.macro_state,
            "is_cash_protection_active": self.is_cash_protection_active,
            "macro_regime_label": self.macro_regime_label,
            "risk_multiplier": self.risk_multiplier
        }


# =====================================================================
# FAZ 23: OTONOM SİNYAL VE FIRSAT MOTORU (AUTONOMOUS OPPORTUNITY ENGINE) ŞEMALARI
# =====================================================================

@dataclass
class PricingTargets:
    current_price: float
    target_price_low: float
    target_price_high: float
    stop_loss_price: float
    expected_return_pct_range: List[float]  # Örn: [12.5, 16.0]
    risk_reward_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_price": round(self.current_price, 2),
            "target_price_low": round(self.target_price_low, 2),
            "target_price_high": round(self.target_price_high, 2),
            "stop_loss_price": round(self.stop_loss_price, 2),
            "expected_return_pct_range": [round(x, 2) for x in self.expected_return_pct_range],
            "risk_reward_ratio": round(self.risk_reward_ratio, 2)
        }

@dataclass
class AIConfidence:
    win_probability_pct: float
    model_verdict: str  # HIGH_CONFLUENCE, MODERATE_CONFLUENCE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "win_probability_pct": round(self.win_probability_pct, 1),
            "model_verdict": self.model_verdict
        }

@dataclass
class AutonomousLearningMetadata:
    evaluation_due_date: str
    status: str                          # PENDING_REALIZATION, REALIZED_SUCCESS, REALIZED_FAILURE
    realized_outcome: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evaluation_due_date": self.evaluation_due_date,
            "status": self.status,
            "realized_outcome": self.realized_outcome
        }

@dataclass
class OpportunityCardOutputData:
    recommendation_id: str
    ticker: str
    timestamp: str
    time_horizon: str                    # 1_WEEK, 1_MONTH
    recommendation_type: str             # STRONGLY_RECOMMEND_BUY, WEEKLY_SCALP_BUY
    pricing_targets: PricingTargets
    ai_confidence: AIConfidence
    key_reasons_shap: List[str]
    autonomous_learning_metadata: AutonomousLearningMetadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommendation_id": self.recommendation_id,
            "ticker": self.ticker,
            "timestamp": self.timestamp,
            "time_horizon": self.time_horizon,
            "recommendation_type": self.recommendation_type,
            "pricing_targets": self.pricing_targets.to_dict(),
            "ai_confidence": self.ai_confidence.to_dict(),
            "key_reasons_shap": self.key_reasons_shap,
            "autonomous_learning_metadata": self.autonomous_learning_metadata.to_dict()
        }


@dataclass
class FibInputData:
    ticker: str
    timeframe: str
    timestamp: str
    pivot_left: int
    pivot_right: int
    ohlcv_data: List[OHLCVBar]
    atr_val: float = 5.4
    confluence_inputs: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FibInputData":
        bars = [OHLCVBar.from_dict(b) for b in data["ohlcv_data"]]
        return cls(
            ticker=data["ticker"],
            timeframe=data.get("timeframe", "4h"),
            timestamp=data["timestamp"],
            pivot_left=int(data.get("pivot_left", 10)),
            pivot_right=int(data.get("pivot_right", 10)),
            ohlcv_data=bars,
            atr_val=float(data.get("atr_val", 5.4)),
            confluence_inputs=data.get("confluence_inputs", {})
        )

@dataclass
class FibFlags:
    price_in_prz_zone: bool = False
    structure_invalidated: bool = False
    high_volume_breakout_threat: bool = False

    def to_dict(self) -> Dict[str, bool]:
        return {
            "price_in_prz_zone": self.price_in_prz_zone,
            "structure_invalidated": self.structure_invalidated,
            "high_volume_breakout_threat": self.high_volume_breakout_threat
        }

@dataclass
class FibOutputData:
    ticker: str
    timeframe: str
    trend_direction: str
    anchors: Dict[str, float]
    fib_levels: Dict[str, Any]
    confluence_analysis: Dict[str, Any]
    flags: Dict[str, bool]
    prz_reversal_score: float
    primary_recommendation_contribution: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "timeframe": self.timeframe,
            "trend_direction": self.trend_direction,
            "anchors": self.anchors,
            "fib_levels": self.fib_levels,
            "confluence_analysis": self.confluence_analysis,
            "flags": self.flags,
            "prz_reversal_score": round(self.prz_reversal_score, 2),
            "primary_recommendation_contribution": self.primary_recommendation_contribution
        }


# =====================================================================
# MODÜL 13: DERİNLİK & EMİR DEFTERİ (ORDER BOOK L2/L3) ŞEMALARI
# =====================================================================

@dataclass
class OrderBookInputData:
    ticker: str
    timestamp: str
    bids: List[Dict[str, float]]
    asks: List[Dict[str, float]]
    cancellations_last_1s: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrderBookInputData":
        return cls(
            ticker=data["ticker"],
            timestamp=data["timestamp"],
            bids=data.get("bids", []),
            asks=data.get("asks", []),
            cancellations_last_1s=int(data.get("cancellations_last_1s", 0))
        )

@dataclass
class OrderBookOutputData:
    ticker: str
    timestamp: str
    obi_ratio: float
    depth_delta_zscore: float
    iceberg_detected: Dict[str, Any]
    spoofing_warning: Dict[str, Any]
    microstructure_signal_score: float
    primary_recommendation_contribution: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp,
            "obi_ratio": round(self.obi_ratio, 2),
            "depth_delta_zscore": round(self.depth_delta_zscore, 2),
            "iceberg_detected": self.iceberg_detected,
            "spoofing_warning": self.spoofing_warning,
            "microstructure_signal_score": round(self.microstructure_signal_score, 2),
            "primary_recommendation_contribution": self.primary_recommendation_contribution
        }


# =====================================================================
# MODÜL 14: ARACI KURUM DAĞILIMI (AKD PROCESSOR) ŞEMALARI
# =====================================================================

@dataclass
class AKDInputData:
    ticker: str
    timestamp: str
    top_buyers: List[Dict[str, Any]]
    top_sellers: List[Dict[str, Any]]
    total_volume: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AKDInputData":
        return cls(
            ticker=data["ticker"],
            timestamp=data["timestamp"],
            top_buyers=data.get("top_buyers", []),
            top_sellers=data.get("top_sellers", []),
            total_volume=float(data.get("total_volume", 1.0))
        )

@dataclass
class AKDOutputData:
    ticker: str
    timestamp: str
    top_5_buyers_share: float
    top_5_sellers_share: float
    dominant_buyer: str
    akd_concentration_score: float
    akd_regime: str
    akd_signal_score: float
    primary_recommendation_contribution: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp,
            "top_5_buyers_share": round(self.top_5_buyers_share, 2),
            "top_5_sellers_share": round(self.top_5_sellers_share, 2),
            "dominant_buyer": self.dominant_buyer,
            "akd_concentration_score": round(self.akd_concentration_score, 2),
            "akd_regime": self.akd_regime,
            "akd_signal_score": round(self.akd_signal_score, 2),
            "primary_recommendation_contribution": self.primary_recommendation_contribution
        }


# =====================================================================
# MODÜL 15: TAKAS VE SAKLAMA ANALİZİ (CUSTODY PROCESSOR) ŞEMALARI
# =====================================================================

@dataclass
class CustodyInputData:
    ticker: str
    timestamp: str
    custody_shares: Dict[str, float]
    weekly_foreign_change: float = 0.0
    total_capital: float = 1.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CustodyInputData":
        return cls(
            ticker=data["ticker"],
            timestamp=data["timestamp"],
            custody_shares=data.get("custody_shares", {}),
            weekly_foreign_change=float(data.get("weekly_foreign_change", 0.0)),
            total_capital=float(data.get("total_capital", 1.0))
        )

@dataclass
class CustodyOutputData:
    ticker: str
    timestamp: str
    top_3_custody_pct: float
    weekly_foreign_custody_change_shares: float
    custody_concentration_index: float
    flags: Dict[str, bool]
    custody_signal_score: float
    primary_recommendation_contribution: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp,
            "top_3_custody_pct": round(self.top_3_custody_pct, 3),
            "weekly_foreign_custody_change_shares": round(self.weekly_foreign_custody_change_shares, 0),
            "custody_concentration_index": round(self.custody_concentration_index, 2),
            "flags": self.flags,
            "custody_signal_score": round(self.custody_signal_score, 2),
            "primary_recommendation_contribution": self.primary_recommendation_contribution
        }


# =====================================================================
# MODÜL 16: NET PARA AKIŞI & FÜZYON (MONEY FLOW PROCESSOR) ŞEMALARI
# =====================================================================

@dataclass
class MoneyFlowInputData:
    ticker: str
    timestamp: str
    order_book_obi: float
    top_5_akd_concentration: float
    weekly_custody_change_pct: float
    net_money_flow_tl: float
    price_change_today_pct: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MoneyFlowInputData":
        return cls(
            ticker=data["ticker"],
            timestamp=data["timestamp"],
            order_book_obi=float(data.get("order_book_obi", 0.0)),
            top_5_akd_concentration=float(data.get("top_5_akd_concentration", 0.0)),
            weekly_custody_change_pct=float(data.get("weekly_custody_change_pct", 0.0)),
            net_money_flow_tl=float(data.get("net_money_flow_tl", 0.0)),
            price_change_today_pct=float(data.get("price_change_today_pct", 0.0))
        )

@dataclass
class MicrostructureFusionOutputData:
    ticker: str
    microstructure_regime: str
    flow_scores: Dict[str, float]
    divergence_flags: Dict[str, bool]
    overall_microstructure_signal: float
    primary_recommendation_contribution: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "microstructure_regime": self.microstructure_regime,
            "flow_scores": {k: round(v, 2) for k, v in self.flow_scores.items()},
            "divergence_flags": self.divergence_flags,
            "overall_microstructure_signal": round(self.overall_microstructure_signal, 2),
            "primary_recommendation_contribution": self.primary_recommendation_contribution
        }


# =====================================================================
# MODÜL 17: BETA KATSAYISI & RİSK (BETA PROCESSOR) ŞEMALARI
# =====================================================================

@dataclass
class AsymmetricBeta:
    up_market_beta: float
    down_market_beta: float
    asymmetry_ratio: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "up_market_beta": round(self.up_market_beta, 2),
            "down_market_beta": round(self.down_market_beta, 2),
            "asymmetry_ratio": round(self.asymmetry_ratio, 3)
        }

@dataclass
class BetaInputData:
    ticker: str
    benchmark_ticker: str
    timestamp: str
    lookback_window_days: int
    raw_beta: float
    stock_returns_std: float
    benchmark_returns_std: float
    correlation: float
    net_debt: float = 0.0
    equity: float = 1.0
    tax_rate: float = 0.25
    illiquidity_flag: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BetaInputData":
        return cls(
            ticker=data["ticker"],
            benchmark_ticker=data.get("benchmark_ticker", "XU100"),
            timestamp=data["timestamp"],
            lookback_window_days=int(data.get("lookback_window_days", 252)),
            raw_beta=float(data["raw_beta"]),
            stock_returns_std=float(data["stock_returns_std"]),
            benchmark_returns_std=float(data["benchmark_returns_std"]),
            correlation=float(data["correlation"]),
            net_debt=float(data.get("net_debt", 0.0)),
            equity=float(data.get("equity", 1.0)),
            tax_rate=float(data.get("tax_rate", 0.25)),
            illiquidity_flag=bool(data.get("illiquidity_flag", False))
        )

@dataclass
class BetaFlags:
    is_high_beta_aggressive: bool = False
    is_thin_trading_biased: bool = False
    favorable_asymmetry_detected: bool = False

    def to_dict(self) -> Dict[str, bool]:
        return {
            "is_high_beta_aggressive": self.is_high_beta_aggressive,
            "is_thin_trading_biased": self.is_thin_trading_biased,
            "favorable_asymmetry_detected": self.favorable_asymmetry_detected
        }

@dataclass
class BetaOutputData:
    ticker: str
    raw_beta: float
    blume_adjusted_beta: float
    unlevered_beta: float
    asymmetric_beta: Dict[str, float]
    flags: Dict[str, bool]
    risk_contribution_score: float
    primary_recommendation_contribution: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "raw_beta": round(self.raw_beta, 2),
            "blume_adjusted_beta": round(self.blume_adjusted_beta, 2),
            "unlevered_beta": round(self.unlevered_beta, 2),
            "asymmetric_beta": self.asymmetric_beta,
            "flags": self.flags,
            "risk_contribution_score": round(self.risk_contribution_score, 2),
            "primary_recommendation_contribution": self.primary_recommendation_contribution
        }


# =====================================================================
# MODÜL 18: SERBEST DOLAŞIM ORANI & KISITLAR (FREE FLOAT PROCESSOR) ŞEMALARI
# =====================================================================

@dataclass
class FreeFloatInputData:
    ticker: str
    timestamp: str
    total_shares_outstanding: float
    free_float_shares: float
    current_price: float
    daily_volume_shares: float = 0.0
    lockup_expiration_date: Optional[str] = None
    lockup_shares_count: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FreeFloatInputData":
        return cls(
            ticker=data["ticker"],
            timestamp=data["timestamp"],
            total_shares_outstanding=float(data["total_shares_outstanding"]),
            free_float_shares=float(data["free_float_shares"]),
            current_price=float(data["current_price"]),
            daily_volume_shares=float(data.get("daily_volume_shares", 0.0)),
            lockup_expiration_date=data.get("lockup_expiration_date"),
            lockup_shares_count=float(data.get("lockup_shares_count", 0.0))
        )

@dataclass
class FreeFloatFlags:
    is_micro_float_danger: bool = False
    lockup_cliff_risk_active: bool = False
    float_rotation_overshoot: bool = False
    trading_execution_blocked: bool = False

    def to_dict(self) -> Dict[str, bool]:
        return {
            "is_micro_float_danger": self.is_micro_float_danger,
            "lockup_cliff_risk_active": self.lockup_cliff_risk_active,
            "float_rotation_overshoot": self.float_rotation_overshoot,
            "trading_execution_blocked": self.trading_execution_blocked
        }

@dataclass
class FreeFloatOutputData:
    ticker: str
    free_float_pct: float
    free_float_market_cap_tl: float
    cornering_risk_index: float
    float_rotation_speed: float
    flags: Dict[str, bool]
    liquidity_risk_score: float
    primary_recommendation_contribution: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "free_float_pct": round(self.free_float_pct, 2),
            "free_float_market_cap_tl": round(self.free_float_market_cap_tl, 2),
            "cornering_risk_index": round(self.cornering_risk_index, 2),
            "float_rotation_speed": round(self.float_rotation_speed, 2),
            "flags": self.flags,
            "liquidity_risk_score": round(self.liquidity_risk_score, 2),
            "primary_recommendation_contribution": self.primary_recommendation_contribution
        }


# =====================================================================
# MODÜL 19: SHARPE ORANI & PERFORMANS (SHARPE PROCESSOR) ŞEMALARI
# =====================================================================

@dataclass
class SharpeInputData:
    asset_id: str
    timestamp: str
    frequency: str
    lookback_periods: int
    asset_returns: List[float]
    risk_free_rate_annual: float
    skewness: float = 0.0
    kurtosis: float = 3.0
    backtest_trials_count: int = 1

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SharpeInputData":
        return cls(
            asset_id=data["asset_id"],
            timestamp=data["timestamp"],
            frequency=data.get("frequency", "DAILY"),
            lookback_periods=int(data.get("lookback_periods", 252)),
            asset_returns=[float(x) for x in data.get("asset_returns", [])],
            risk_free_rate_annual=float(data["risk_free_rate_annual"]),
            skewness=float(data.get("skewness", 0.0)),
            kurtosis=float(data.get("kurtosis", 3.0)),
            backtest_trials_count=int(data.get("backtest_trials_count", 1))
        )

@dataclass
class SharpeFlags:
    is_gamed_or_smoothed: bool = False
    negative_skewness_tail_risk: bool = False
    exceeds_risk_free_significantly: bool = True
    overfitting_rejected: bool = False

    def to_dict(self) -> Dict[str, bool]:
        return {
            "is_gamed_or_smoothed": self.is_gamed_or_smoothed,
            "negative_skewness_tail_risk": self.negative_skewness_tail_risk,
            "exceeds_risk_free_significantly": self.exceeds_risk_free_significantly,
            "overfitting_rejected": self.overfitting_rejected
        }

@dataclass
class SharpeOutputData:
    asset_id: str
    raw_annualized_sharpe: float
    lo_autocorr_adjusted_sharpe: float
    favre_galeano_adjusted_sharpe: float
    deflated_sharpe_ratio_dsr: float
    flags: Dict[str, bool]
    final_risk_adjusted_score: float
    primary_recommendation_contribution: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "raw_annualized_sharpe": round(self.raw_annualized_sharpe, 2),
            "lo_autocorr_adjusted_sharpe": round(self.lo_autocorr_adjusted_sharpe, 2),
            "favre_galeano_adjusted_sharpe": round(self.favre_galeano_adjusted_sharpe, 2),
            "deflated_sharpe_ratio_dsr": round(self.deflated_sharpe_ratio_dsr, 2),
            "flags": self.flags,
            "final_risk_adjusted_score": round(self.final_risk_adjusted_score, 2),
            "primary_recommendation_contribution": self.primary_recommendation_contribution
        }


# =====================================================================
# FAZ 20: ÇOK FAKTÖRLÜ KARAR FÜZYON MOTORU (MASTER DECISION FUSION) ŞEMALARI
# =====================================================================

@dataclass
class MasterFusionInputData:
    ticker: str
    timestamp: str
    pe_output: Dict[str, Any]
    pb_output: Dict[str, Any]
    ebitda_output: Dict[str, Any]
    ev_output: Dict[str, Any]
    roe_output: Dict[str, Any]
    current_ratio_output: Dict[str, Any]
    rsi_output: Dict[str, Any]
    ma_output: Dict[str, Any]
    macd_output: Dict[str, Any]
    bollinger_output: Dict[str, Any]
    volume_output: Dict[str, Any]
    fib_output: Dict[str, Any]
    order_book_output: Dict[str, Any]
    akd_output: Dict[str, Any]
    custody_output: Dict[str, Any]
    microstructure_fusion_output: Dict[str, Any]
    beta_output: Dict[str, Any]
    free_float_output: Dict[str, Any]
    sharpe_output: Dict[str, Any]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MasterFusionInputData":
        return cls(
            ticker=data["ticker"],
            timestamp=data["timestamp"],
            pe_output=data.get("pe_output", {}),
            pb_output=data.get("pb_output", {}),
            ebitda_output=data.get("ebitda_output", {}),
            ev_output=data.get("ev_output", {}),
            roe_output=data.get("roe_output", {}),
            current_ratio_output=data.get("current_ratio_output", {}),
            rsi_output=data.get("rsi_output", {}),
            ma_output=data.get("ma_output", {}),
            macd_output=data.get("macd_output", {}),
            bollinger_output=data.get("bollinger_output", {}),
            volume_output=data.get("volume_output", {}),
            fib_output=data.get("fib_output", {}),
            order_book_output=data.get("order_book_output", {}),
            akd_output=data.get("akd_output", {}),
            custody_output=data.get("custody_output", {}),
            microstructure_fusion_output=data.get("microstructure_fusion_output", {}),
            beta_output=data.get("beta_output", {}),
            free_float_output=data.get("free_float_output", {}),
            sharpe_output=data.get("sharpe_output", {})
        )

@dataclass
class SubSystemScores:
    fundamental_score: float
    technical_score: float
    microstructure_score: float
    risk_score: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "fundamental_score": round(self.fundamental_score, 2),
            "technical_score": round(self.technical_score, 2),
            "microstructure_score": round(self.microstructure_score, 2),
            "risk_score": round(self.risk_score, 2)
        }

@dataclass
class MasterFusionOutputData:
    ticker: str
    timestamp: str
    composite_master_score: float
    final_recommendation: str
    subsystem_scores: Dict[str, float]
    positive_drivers: List[str]
    critical_warnings: List[str]
    hard_safety_block_triggered: bool
    recommended_portfolio_weight_pct: float
    max_position_cap_shares: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp,
            "composite_master_score": round(self.composite_master_score, 2),
            "final_recommendation": self.final_recommendation,
            "subsystem_scores": self.subsystem_scores,
            "positive_drivers": self.positive_drivers,
            "critical_warnings": self.critical_warnings,
            "hard_safety_block_triggered": self.hard_safety_block_triggered,
            "recommended_portfolio_weight_pct": round(self.recommended_portfolio_weight_pct, 2),
            "max_position_cap_shares": round(self.max_position_cap_shares, 0)
        }


# =====================================================================
# FAZ 21: KURUMSAL QUANT YÜKSELTME (INSTITUTIONAL QUANT UPGRADE) ŞEMALARI
# =====================================================================

@dataclass
class BiTemporalTimestamp:
    event_timestamp: str       # t_event (T anı)
    knowledge_timestamp: str   # t_knowledge (T+2 / T+1 kamuya açıklanma anı)

    def is_valid_at(self, current_sim_time: str) -> bool:
        return self.knowledge_timestamp <= current_sim_time

@dataclass
class MetaLabelingInputData:
    ticker: str
    timestamp: str
    primary_signal_side: int  # -1, 0, +1
    volatility_atr: float
    bid_ask_spread: float
    order_book_imbalance_obi: float
    volume_intensity: float
    historical_win_rate: float = 0.55

@dataclass
class MetaLabelingOutputData:
    ticker: str
    p_success: float          # P(Success)
    p_size: float             # P_size = 2 * P(Success) - 1
    is_meta_approved: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "p_success": round(self.p_success, 4),
            "p_size": round(self.p_size, 4),
            "is_meta_approved": self.is_meta_approved
        }

@dataclass
class MarketImpactInputData:
    ticker: str
    order_volume_shares: float
    adv_20_shares: float
    daily_volatility: float
    lower_barrier_price: float
    is_limit_down_locked: bool = False
    gamma: float = 0.1
    alpha: float = 0.5

@dataclass
class MarketImpactOutputData:
    ticker: str
    executed_price: float
    slippage_pct: float
    execution_feasible: bool
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "executed_price": round(self.executed_price, 2),
            "slippage_pct": round(self.slippage_pct, 4),
            "execution_feasible": self.execution_feasible,
            "rejection_reason": self.rejection_reason
        }

@dataclass
class SampleUniquenessInputData:
    label_start_times: List[int]
    label_end_times: List[int]
    returns: List[float]

@dataclass
class SampleUniquenessOutputData:
    mean_uniqueness_scores: List[float]
    sample_weights: List[float]

