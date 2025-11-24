# linear interpolation code

import numpy as np

def linear_edge_samples(qa: np.ndarray, qb: np.ndarray, res: float):
    # qa -> qb 구간 선형 보간 
    # res: edge 샘플 구간
    # 반환: qa 제외, qb 포함한 샘플들을 yield

    qa = np.asarray(qa, dtype=float)
    qb = np.asarray(qb, dtype=float)
    seg_len = np.linalg.norm(qb - qa)
    if seg_len == 0.0:
        yield qb
        return
    
    n = max(1, int(np.ceil(seg_len / res)))
    for k in range(1, n + 1):
        t = k / n
        yield qa * (1.0 - t) + qb * t