# 生成 Discussion/PR 配图（ES-MoE 实战项目一）
# 输出到本脚本所在目录 figures/（含热力图复制 + 3 张新图）
import os, csv, shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

fig_dir = os.path.dirname(os.path.abspath(__file__))

# 1. 复制 K=2 路由热力图（主图）
src = r'g:/Codes/OpenSource/Rhino-bird/practices/Codes copy 3/YOLO-Master/experiments_zviolin/runs/esmoe_routing_k2/routing_heatmap.png'
shutil.copy(src, os.path.join(fig_dir, 'routing_heatmap_k2.png'))

# 2. 路由熵对比 + 专家利用率堆叠
variants = ['K=1', 'K=2\n(baseline)', 'K=3', 'Shared']
entropy = [0.198, 0.490, 0.741, 0.430]
util = [
    [0.250, 0.750, 0.000, 0.000],
    [0.209, 0.241, 0.215, 0.335],
    [0.269, 0.218, 0.190, 0.322],
    [0.257, 0.106, 0.214, 0.423],
]
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
labels = ['Expert0 (3x3)', 'Expert1 (5x5)', 'Expert2 (7x7)', 'Expert3 (9x9)']
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].bar(variants, entropy, color=['#4C72B0', '#DD8452', '#55A868', '#C44E52'])
ax[0].set_title('Routing Entropy $H_{norm}$ (higher = more uniform)')
ax[0].set_ylim(0, 1)
for i, v in enumerate(entropy):
    ax[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
bottom = np.zeros(4)
for j in range(4):
    vals = [u[j] for u in util]
    ax[1].bar(variants, vals, bottom=bottom, color=colors[j], label=labels[j], width=0.6)
    bottom += np.array(vals)
ax[1].set_title('Expert Utilization (mean weight, 4 ES_MOE layers)')
ax[1].legend(fontsize=8, loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'routing_entropy_compare.png'), dpi=150)
plt.close()

# 3. 稀疏化收益（同权重 Hard Top-K vs 强制 Dense）
gpu = {'K=1': (27.341, 35.853), 'K=3': (31.118, 34.780), 'Shared': (29.202, 33.933)}
cpu = {'K=1': (56.658, 69.900), 'K=2': (60.687, 69.970), 'K=3': (58.254, 69.126), 'Shared': (65.559, 70.513)}
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, data, title in [(axes[0], gpu, 'GPU (cuda:0)'), (axes[1], cpu, 'CPU')]:
    ks = list(data.keys())
    hard = [data[k][0] for k in ks]
    dense = [data[k][1] for k in ks]
    x = np.arange(len(ks))
    w = 0.35
    ax.bar(x - w / 2, hard, w, label='Hard Top-K (sparse)', color='#55A868')
    ax.bar(x + w / 2, dense, w, label='Dense (forced, 4/4)', color='#4C72B0')
    for i, (h, d) in enumerate(zip(hard, dense)):
        gain = (h - d) / d * 100
        ax.text(i, max(h, d) + 1, f'{h:.1f}ms\n({gain:+.1f}%)', ha='center', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(ks)
    ax.set_title(f'Inference Latency: {title}')
    ax.set_ylabel('ms/frame')
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'sparse_gain_gpu_cpu.png'), dpi=150)
plt.close()

# 4. 动态 Top-K：场景复杂度与专家选择关联
csv_path = r'g:/Codes/OpenSource/Rhino-bird/practices/Codes copy 3/YOLO-Master/experiments_zviolin/runs/dynamic_topk_k2/dynamic_topk_per_image.csv'
comp, tk = [], []
with open(csv_path, encoding='utf-8') as f:
    for row in csv.DictReader(f):
        comp.append(float(row['complexity']))
        tk.append(int(row['top_k']))
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(comp, tk, alpha=0.6, s=45, color='#C44E52', edgecolor='white', linewidth=0.5)
ax.axhline(1.5, color='gray', ls='--', lw=0.8)
ax.set_xlabel('Image complexity (edge intensity, L1)')
ax.set_ylabel('Selected top_k')
ax.set_title(f'Dynamic Top-K Router (avg top_k={np.mean(tk):.2f}, n={len(tk)})')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'dynamic_topk_complexity.png'), dpi=150)
plt.close()

print('OK, figures:', sorted(os.listdir(fig_dir)))
