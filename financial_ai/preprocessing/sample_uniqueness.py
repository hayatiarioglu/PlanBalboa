import numpy as np
from typing import List
from financial_ai.schemas import SampleUniquenessInputData, SampleUniquenessOutputData

class SampleUniquenessEngine:
    """
    FAZ 21 DÜZELTME 2: Örtüşen Etiketler İçin Marcos López de Prado Sample Uniqueness (u_i) Matematiği.
    Üçlü Engel (Triple Barrier) etiketlerindeki zaman serisi çakışmalarını (Concurrency) hesaplayarak
    XGBoost/LightGBM modellerindeki ezberleme (Overfitting) riskini ortadan kaldırır.
    """

    def compute_uniqueness(self, input_data: SampleUniquenessInputData) -> SampleUniquenessOutputData:
        starts = input_data.label_start_times
        ends = input_data.label_end_times
        returns = input_data.returns
        n = len(starts)

        if n == 0:
            return SampleUniquenessOutputData(mean_uniqueness_scores=[], sample_weights=[])

        max_t = max(ends) if ends else 100
        concurrency = np.zeros(max_t + 1, dtype=float)

        # 1. Compute Concurrency c_t
        for i in range(n):
            concurrency[starts[i]:ends[i] + 1] += 1.0

        u_i_list: List[float] = []
        w_i_list: List[float] = []

        # 2. Compute Point Uniqueness u_{i,t} and Mean Uniqueness u_i
        for i in range(n):
            t_span = range(starts[i], ends[i] + 1)
            length = len(t_span)
            u_it = [1.0 / concurrency[t] for t in t_span]
            u_i = float(np.mean(u_it)) if length > 0 else 1.0
            u_i_list.append(round(u_i, 4))

            # Sample weight: w_i = u_i * |r_i|
            r_abs = abs(returns[i]) if i < len(returns) else 1.0
            w_i = u_i * r_abs
            w_i_list.append(round(w_i, 4))

        return SampleUniquenessOutputData(
            mean_uniqueness_scores=u_i_list,
            sample_weights=w_i_list
        )
