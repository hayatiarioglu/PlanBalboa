"""
AetherForecaster-X v35.2 Multi-Asset Master
Katman 3: Valör-Cezalı Bounded Convex QP Solver

Modül: OSQP Solver Wrapper (Operator Splitting Quadratic Program Çözücü Arayüzü)
Faz 2.4 / Adım 2.4.1: Hessian (P), Alpha Vektörü (q), Kısıt Matrisi (A) ve Sınırlar (l, u)
kullanılarak OSQP konveks optimizasyonunun güvenli çalıştırılması.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union
import warnings
import numpy as np
import pandas as pd
from scipy import sparse
import osqp


@dataclass
class OSQPResult:
    """
    OSQP Konveks Çözücünün Niteleyici Çıktı Paketi.
    """
    x: np.ndarray                   # Optimal çözüm vektörü (ağırlıklar / değişimler)
    status: str                     # Çözüm durumu ('solved', 'solved_inaccurate', vb.)
    status_val: int                 # OSQP durum kodu
    obj_val: float                  # Amaç fonksiyonu değeri (1/2 x^T P x + q^T x)
    run_time: float                 # Çözüm süresi (saniye)
    is_success: bool                # Başarı durumu (solved veya solved_inaccurate ise True)
    info: Dict[str, Any] = field(default_factory=dict)


class OSQPSolverWrapper:
    """
    Aether Katman 3 konveks optimizasyonu için OSQP (Operator Splitting Quadratic Program)
    çözücü arayüzü ve güvenlik katmanı.
    
    Matematiksel Formülasyon:
        min_x  1/2 * x^T * P * x + q^T * x
        s.t.   l <= A * x <= u
    """

    SUCCESS_STATUSES = {"solved", "solved_inaccurate"}

    def __init__(
        self,
        eps_abs: float = 1e-5,
        eps_rel: float = 1e-5,
        max_iter: int = 10000,
        verbose: bool = False,
        polish: bool = True,
        adaptive_rho: bool = True
    ):
        """
        :param eps_abs: Mutlak yakınsama toleransı.
        :param eps_rel: Göreli yakınsama toleransı.
        :param max_iter: Maksimum iterasyon sayısı.
        :param verbose: Çözücü loglarının konsola yazdırılması.
        :param polish: Çözümün son hassaslaştırma (polishing) adımı.
        :param adaptive_rho: Otomatik step-size (rho) adaptasyonu.
        """
        self.eps_abs = eps_abs
        self.eps_rel = eps_rel
        self.max_iter = max_iter
        self.verbose = verbose
        self.polish = polish
        self.adaptive_rho = adaptive_rho
        self._solver: Optional[osqp.OSQP] = None
        self._is_setup: bool = False

    def _to_numpy_array(self, val: Union[np.ndarray, pd.DataFrame, pd.Series, sparse.spmatrix]) -> np.ndarray:
        """Farklı veri tiplerini güvenli şekilde dense numpy array'e dönüştürür."""
        if isinstance(val, (pd.DataFrame, pd.Series)):
            return val.to_numpy(dtype=np.float64)
        if sparse.issparse(val):
            return val.toarray().astype(np.float64)
        return np.asarray(val, dtype=np.float64)

    def setup(
        self,
        P: Union[np.ndarray, pd.DataFrame, sparse.spmatrix],
        q: Union[np.ndarray, pd.Series],
        A: Union[np.ndarray, pd.DataFrame, sparse.spmatrix],
        l: Union[np.ndarray, pd.Series],
        u: Union[np.ndarray, pd.Series]
    ) -> None:
        """
        Faz 2.4.4: OSQP çözücü modelini persistent olarak hazırlar ve RAM üzerinde yapılandırır.
        """
        P_arr = self._to_numpy_array(P)
        q_arr = self._to_numpy_array(q).flatten()
        A_arr = self._to_numpy_array(A)
        l_arr = self._to_numpy_array(l).flatten()
        u_arr = self._to_numpy_array(u).flatten()

        if not (np.all(np.isfinite(P_arr)) and np.all(np.isfinite(q_arr)) and np.all(np.isfinite(A_arr))):
            raise ValueError("OSQP P, q, A matrislerinde NaN veya Inf tespit edildi!")

        if np.any(np.isnan(l_arr)) or np.any(np.isnan(u_arr)):
            raise ValueError("OSQP l veya u sınır vektörlerinde NaN tespit edildi!")

        n_p, m_p = P_arr.shape
        if n_p != m_p:
            raise ValueError(f"Hessian P matrisi kare olmalıdır! Boyut: {P_arr.shape}")

        n_assets = n_p
        if len(q_arr) != n_assets:
            raise ValueError(f"Alpha vektörü q boyutu ({len(q_arr)}), P matrisi boyutu ({n_assets}) ile uyuşmuyor!")

        m_constraints, n_vars = A_arr.shape
        if n_vars != n_assets:
            raise ValueError(f"Kısıt matrisi A değişken sayısı ({n_vars}), P boyutu ({n_assets}) ile uyuşmuyor!")

        if len(l_arr) != m_constraints or len(u_arr) != m_constraints:
            raise ValueError(f"Sınır vektörleri l/u boyutları ({len(l_arr)}/{len(u_arr)}), A kısıt sayısı ({m_constraints}) ile uyuşmuyor!")

        if np.any(l_arr > u_arr + 1e-12):
            raise ValueError("Kısıt alt sınırı (l), üst sınırından (u) büyük olamaz!")

        P_triu = np.triu(P_arr)
        P_sparse = sparse.csc_matrix(P_triu)
        A_sparse = sparse.csc_matrix(A_arr)

        self._solver = osqp.OSQP()
        setup_kwargs = {
            "P": P_sparse,
            "q": q_arr,
            "A": A_sparse,
            "l": l_arr,
            "u": u_arr,
            "eps_abs": self.eps_abs,
            "eps_rel": self.eps_rel,
            "max_iter": self.max_iter,
            "verbose": self.verbose,
            "adaptive_rho": self.adaptive_rho
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            try:
                self._solver.setup(**setup_kwargs, polishing=self.polish)
            except (TypeError, ValueError):
                self._solver.setup(**setup_kwargs, polish=self.polish)

        self._is_setup = True

    def update_settings(
        self,
        eps_abs: Optional[float] = None,
        eps_rel: Optional[float] = None,
        max_iter: Optional[int] = None,
        verbose: Optional[bool] = None
    ) -> None:
        """
        Faz 2.4.4: Çözücü hassasiyet toleranslarını (eps_abs, eps_rel) ve maksimum iterasyon
        sayısını aktif OSQP örneği üzerinde dinamik günceller.
        """
        if not self._is_setup or self._solver is None:
            raise RuntimeError("OSQP çözücü henüz setup edilmedi! Önce setup() metodunu çağırın.")

        kwargs = {}
        if eps_abs is not None:
            self.eps_abs = eps_abs
            kwargs["eps_abs"] = eps_abs
        if eps_rel is not None:
            self.eps_rel = eps_rel
            kwargs["eps_rel"] = eps_rel
        if max_iter is not None:
            self.max_iter = max_iter
            kwargs["max_iter"] = max_iter
        if verbose is not None:
            self.verbose = verbose
            kwargs["verbose"] = verbose

        if kwargs:
            self._solver.update_settings(**kwargs)

    def update_vectors(
        self,
        q: Optional[Union[np.ndarray, pd.Series]] = None,
        l: Optional[Union[np.ndarray, pd.Series]] = None,
        u: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> None:
        """
        Faz 2.4.4: Sıfırdan matris kurmadan alpha sinyal vektörü (q) ve kısıt sınırlarını (l, u)
        hızlıca günceller.
        """
        if not self._is_setup or self._solver is None:
            raise RuntimeError("OSQP çözücü henüz setup edilmedi!")

        kwargs = {}
        if q is not None:
            kwargs["q"] = self._to_numpy_array(q).flatten()
        if l is not None:
            kwargs["l"] = self._to_numpy_array(l).flatten()
        if u is not None:
            kwargs["u"] = self._to_numpy_array(u).flatten()

        if kwargs:
            self._solver.update(**kwargs)

    def warm_start(
        self,
        x: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None
    ) -> None:
        """
        Faz 2.4.4: Başlangıç çözüm noktasını (Primal x / Dual y) warm-start ile çözücüye besler.
        """
        if not self._is_setup or self._solver is None:
            raise RuntimeError("OSQP çözücü henüz setup edilmedi!")

        kwargs = {}
        if x is not None:
            kwargs["x"] = self._to_numpy_array(x).flatten()
        if y is not None:
            kwargs["y"] = self._to_numpy_array(y).flatten()

        if kwargs:
            self._solver.warm_start(**kwargs)

    def solve_prepared(self) -> OSQPResult:
        """
        Faz 2.4.4: Setup/Update edilmiş persistent OSQP modelini çalıştırır.
        """
        if not self._is_setup or self._solver is None:
            raise RuntimeError("OSQP çözücü henüz setup edilmedi!")

        res = self._solver.solve()

        status_str = str(res.info.status)
        is_success = status_str in self.SUCCESS_STATUSES
        n_vars = res.x.shape[0] if res.x is not None else 0
        x_sol = res.x if res.x is not None else np.full(n_vars, np.nan)
        obj_val = float(res.info.obj_val) if hasattr(res.info, "obj_val") and res.info.obj_val is not None else float("nan")
        run_time = float(res.info.run_time) if hasattr(res.info, "run_time") and res.info.run_time is not None else 0.0

        info_dict = {
            "iter": int(res.info.iter),
            "status_val": int(res.info.status_val),
            "status_polish": getattr(res.info, "status_polish", None),
            "pri_res": float(res.info.pri_res) if hasattr(res.info, "pri_res") else None,
            "dua_res": float(res.info.dua_res) if hasattr(res.info, "dua_res") else None
        }

        return OSQPResult(
            x=x_sol,
            status=status_str,
            status_val=int(res.info.status_val),
            obj_val=obj_val,
            run_time=run_time,
            is_success=is_success,
            info=info_dict
        )

    def solve(
        self,
        P: Union[np.ndarray, pd.DataFrame, sparse.spmatrix],
        q: Union[np.ndarray, pd.Series],
        A: Union[np.ndarray, pd.DataFrame, sparse.spmatrix],
        l: Union[np.ndarray, pd.Series],
        u: Union[np.ndarray, pd.Series]
    ) -> OSQPResult:
        """
        OSQP çözücüsünü tek adımda hazırlar (setup), çalıştırır ve sonucu döndürür.
        """
        self.setup(P, q, A, l, u)
        return self.solve_prepared()

    def solve_with_adaptive_tolerance(
        self,
        P: Union[np.ndarray, pd.DataFrame, sparse.spmatrix],
        q: Union[np.ndarray, pd.Series],
        A: Union[np.ndarray, pd.DataFrame, sparse.spmatrix],
        l: Union[np.ndarray, pd.Series],
        u: Union[np.ndarray, pd.Series],
        target_eps_abs: float = 1e-6,
        target_eps_rel: float = 1e-6,
        fallback_eps_abs: float = 1e-4,
        fallback_eps_rel: float = 1e-4
    ) -> OSQPResult:
        """
        Faz 2.4.4: Adaptif Tolerans Entegrasyonu.
        Önce hedef dar toleransla (eps_abs, eps_rel) çözer. Yakınsama başarısız olursa,
        eps_abs ve eps_rel toleranslarını esneterek kademeli olarak (fallback) dener.
        """
        self.eps_abs = target_eps_abs
        self.eps_rel = target_eps_rel
        self.setup(P, q, A, l, u)
        res = self.solve_prepared()

        if res.is_success:
            return res

        # Dar toleransta konverje olamadıysa esnek toleransa geç
        self.update_settings(eps_abs=fallback_eps_abs, eps_rel=fallback_eps_rel)
        res_fallback = self.solve_prepared()

        return res_fallback

