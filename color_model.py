import numpy as np

# sRGB <-> linear
def srgb_to_linear(arr: np.ndarray) -> np.ndarray:
    arr = np.clip(arr, 0.0, 1.0)
    mask = arr <= 0.04045
    out = np.empty_like(arr)
    out[mask] = arr[mask] / 12.92
    out[~mask] = ((arr[~mask] + 0.055) / 1.055) ** 2.4
    return out


def linear_to_srgb(arr: np.ndarray) -> np.ndarray:
    arr = np.clip(arr, 0.0, 1.0)
    mask = arr <= 0.0031308
    out = np.empty_like(arr)
    out[mask] = arr[mask] * 12.92
    out[~mask] = 1.055 * (arr[~mask] ** (1 / 2.4)) - 0.055
    return out


# 3x6 흡수 매트릭스: 각 채널이 R/G/B에 얼마나 빛을 흡수하는지 정의 (0~1)
# 초기값을 보다 직교(순수 CMY)하게 조정해 채널 간 크로스톡을 줄였습니다.
DEFAULT_ABSORBANCE = np.array(
    [
        # Cyan  Magenta  Yellow  Black  Red   Gray
        [0.95, 0.00, 0.00, 0.70, 0.30, 0.33],  # R 흡수 (Red 감소)
        [0.00, 0.95, 0.00, 0.70, 0.55, 0.33],  # G 흡수 (Green 감소)
        [0.00, 0.00, 0.95, 0.70, 0.55, 0.33],  # B 흡수 (Blue 감소)
    ],
    dtype=np.float32,
)

# 사전 계산된 의사역행렬 (6x3) for DEFAULT_ABSORBANCE
DEFAULT_PINV = np.linalg.pinv(DEFAULT_ABSORBANCE)


def compute_pinv(absorb: np.ndarray) -> np.ndarray:
    """흡수 매트릭스(3x6)의 의사역행렬을 계산."""
    return np.linalg.pinv(absorb)


def channels_from_rgb_linear(
    rgb_lin: np.ndarray,
    absorb: np.ndarray = DEFAULT_ABSORBANCE,
    pinv: np.ndarray = DEFAULT_PINV,
) -> np.ndarray:
    """선형 RGB(0~1)를 6채널 흡수량(0~1)으로 변환."""
    target = 1.0 - rgb_lin  # 흡수해야 할 양
    flat = target.reshape(-1, 3).T  # (3, N)
    ch = pinv @ flat  # (6, N)
    ch = np.clip(ch, 0.0, 1.0)
    return ch.T.reshape(rgb_lin.shape[0], rgb_lin.shape[1], 6)


def rgb_linear_from_channels(
    channels: np.ndarray, absorb: np.ndarray = DEFAULT_ABSORBANCE
) -> np.ndarray:
    """6채널(0~1)로부터 선형 RGB(0~1) 재구성."""
    h, w, _ = channels.shape
    flat_ch = channels.reshape(-1, 6).T  # (6, N)
    absorb_effect = absorb @ flat_ch  # (3, N)
    rgb_lin = 1.0 - absorb_effect
    rgb_lin = np.clip(rgb_lin, 0.0, 1.0)
    return rgb_lin.T.reshape(h, w, 3)
