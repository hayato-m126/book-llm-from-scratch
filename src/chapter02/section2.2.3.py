import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    attention_weights = softmax(np.matmul(Q, K.T))
    return np.matmul(attention_weights, V), attention_weights

x1 = np.random.randn(100) * 10
x2 = np.random.randn(100)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.scatter(x1, softmax(x1), s=3)
ax2.scatter(x2, softmax(x2), s=3)
# ax1.set_ylim(-0.001, 0.1)
ax2.set_ylim(-0.001, 0.1)
ax1.set_xlabel("入力")
ax2.set_xlabel("入力")
ax1.set_ylabel("出力")

fig.savefig("softmax_different_ranges.png", dpi=144)

dim1 = 1
dim2 = 100
n_keys = 20

q1 = np.random.randn(dim1)
q2 = np.random.randn(dim2)
k1 = np.random.randn(n_keys, dim1)
k2 = np.random.randn(n_keys, dim2)

dot1 = np.matmul(k1, q1)
s1_1 = softmax(dot1)
s1_2 = softmax(dot1 / np.sqrt(dim1))

dot2 = np.matmul(k2, q2)
s2_1 = softmax(dot2)
s2_2 = softmax(dot2 / np.sqrt(dim2))

fig, ax = plt.subplots(2, 2, figsize=(12, 12))

max_1_2 = max(np.max(s1_2), np.max(s2_2))
ax[0, 0].scatter(dot1, s1_1, s=5)
ax[0, 0].set_title(f"{dim1} 次元, スケーリングなし")
ax[0, 0].set_ylim(-0.01, max_1_2 + 0.1)
ax[0, 1].scatter(dot1, s1_2, s=5)
ax[0, 1].set_title(f"{dim1} 次元, スケーリングあり")
ax[0, 1].set_ylim(-0.01, max_1_2 + 0.1)
ax[1, 0].scatter(dot2, s2_1, s=5)
ax[1, 0].set_title(f"{dim2} 次元, スケーリングなし")
ax[1, 1].scatter(dot2, s2_2, s=5)
ax[1, 1].set_title(f"{dim2} 次元, スケーリングあり")
ax[1, 1].set_ylim(-0.01, max_1_2 + 0.1)

fig.savefig("softmax_scaling.png", dpi=144)
