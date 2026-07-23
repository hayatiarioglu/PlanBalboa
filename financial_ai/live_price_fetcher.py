import logging
import yfinance as yf
from typing import Dict, Optional

logger = logging.getLogger("LivePriceFetcher")

# Fallback nominal price map for offline or fallback scenarios
NOMINAL_PRICE_MAP: Dict[str, float] = {
    "THYAO": 315.0, "GARAN": 115.0, "ASELS": 65.0, "EREGL": 48.0, "SISE": 45.0,
    "BIMAS": 550.0, "KCHOL": 210.0, "ARCLK": 155.0, "TUPRS": 165.0, "AKBNK": 60.0,
    "YKBNK": 30.0, "SAHOL": 95.0, "PETKM": 19.0, "KOZAL": 22.0, "PGSUS": 230.0,
    "ISCTR": 13.5, "HEKTS": 13.0, "SASA": 38.0, "VAKBN": 18.0, "HALKB": 17.0,
    "TCELL": 95.0, "TTKOM": 52.0, "EKGYO": 11.0, "TOASO": 240.0, "FROTO": 1050.0,
    "ENKAI": 42.0, "GUBRF": 180.0, "ODAS": 7.5, "KONTR": 48.0, "SMRTG": 50.0,
    "EUPWR": 85.0, "KBORU": 65.0, "ASTOR": 95.0, "ALARK": 105.0, "AEFES": 210.0,
    "MAVI": 115.0, "SOKM": 55.0, "AGHOL": 340.0, "DOAS": 260.0, "MGROS": 520.0,
    "BRSAN": 510.0, "CANTE": 17.0, "CEMTS": 11.0, "CIMSA": 35.0, "DOHOL": 14.0,
    "ECILC": 52.0, "EGEEN": 11500.0, "ENJSA": 62.0, "GESAN": 50.0, "GSDHO": 4.5,
    "INVEO": 10.0, "INVES": 320.0, "ISDMR": 38.0, "ISGYO": 16.0, "ISMEN": 36.0,
    "KCAER": 45.0, "KORDS": 85.0, "KOZAA": 55.0, "KRDMD": 25.0, "KZBGY": 18.0,
    "MIATK": 65.0, "MOBTL": 5.5, "MPARK": 320.0, "OTKAR": 480.0, "OYAKC": 58.0,
    "PENTAG": 18.0, "QUAGR": 4.5, "REEDR": 35.0, "SDTTR": 280.0, "SKBNK": 5.5,
    "TATEN": 35.0, "TAVHL": 240.0, "TKFEN": 48.0, "TMSN": 110.0, "TRGYO": 35.0,
    "TSKB": 11.0, "TUKAS": 7.0, "TURSG": 52.0, "ULKER": 150.0, "VESBE": 22.0,
    "VESTL": 80.0, "YEOTK": 210.0, "YYLGD": 14.0, "ZOREN": 5.0, "ALFAS": 85.0,
    "ANSGR": 95.0, "BERA": 16.0, "BFREN": 850.0, "BIENY": 35.0, "BOBET": 38.0,
    "BRYAT": 2400.0, "CWENE": 210.0, "EGPRO": 210.0, "EBEBK": 45.0, "GWIND": 28.0,
    "IMASM": 14.0, "KAYSE": 35.0, "KMPUR": 65.0, "TABGD": 165.0
}

_PRICE_CACHE: Dict[str, float] = {}

def get_live_bist_price(ticker: str, fallback_price: Optional[float] = None) -> float:
    """
    Borsa İstanbul Canlı Fiyat Çekici (yfinance + Cache + Fallback).
    1. yfinance (Yahoo Finance) üzerinden BIST canlı tahta fiyatını çeker (örn: ASELS.IS -> 67.50).
    2. Ağ hatası veya borsa kapalıysa önbellek / NOMINAL_PRICE_MAP haritasını kullanır.
    """
    clean_ticker = ticker.strip().upper()
    if clean_ticker in _PRICE_CACHE:
        return _PRICE_CACHE[clean_ticker]

    yf_symbol = f"{clean_ticker}.IS"
    try:
        t = yf.Ticker(yf_symbol)
        info = getattr(t, "fast_info", None)
        live_p = None
        if info:
            live_p = getattr(info, "last_price", None) or getattr(info, "previous_close", None)
        
        if live_p and float(live_p) > 0:
            price_val = float(live_p)
            _PRICE_CACHE[clean_ticker] = price_val
            logger.info(f"✅ [LIVE PRICE] {clean_ticker}.IS Canlı BIST Fiyatı Çekildi: {price_val:.2f} TL")
            return price_val
    except Exception as e:
        logger.warning(f"[LIVE PRICE WARNING] {clean_ticker} yfinance canlı fiyat çekilemedi: {e}")

    # Fallback to NOMINAL_PRICE_MAP or fallback_price
    fallback_val = NOMINAL_PRICE_MAP.get(clean_ticker, fallback_price or 100.0)
    _PRICE_CACHE[clean_ticker] = fallback_val
    return fallback_val
