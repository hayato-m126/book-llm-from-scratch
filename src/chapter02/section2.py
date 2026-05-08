import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import math


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    attention_weights = softmax(np.matmul(Q, K.T))
    return np.matmul(attention_weights, V), attention_weights


def plot_vectors(vectors, texts, ax, arrow_args={}, text_args={}):
    for i, (x, y) in enumerate(vectors):
        ax.annotate("", xy=[x, y], xytext=[0, 0], arrowprops=arrow_args)
        ax.text(x * 1.1, y * 1.1, texts[i], **text_args)


def plot_attention_hist(attention_weights, ax):
    ax.bar(range(len(attention_weights)), attention_weights)
    ax.set_xticks(range(len(attention_weights)))
    ax.set_xticklabels([f"$v_{i+1}$" for i in range(len(attention_weights))])
    ax.set_ylabel("注意度")

n = 10
vectors = []
# 360°をn等分した角度
theta = 2 * math.pi / n
for i in range(n):
    x = math.cos(theta * i)
    y = math.sin(theta * i)
    vectors.append([x, y])

vectors = np.array(vectors)
query = np.array([1 / math.sqrt(2), 1 / math.sqrt(2)])
output, attention_weights = attention(query, vectors, vectors)
print("出力ベクトル:", output)
print("注意度の形:", attention_weights.shape)
print("注意度の和:", attention_weights.sum())
