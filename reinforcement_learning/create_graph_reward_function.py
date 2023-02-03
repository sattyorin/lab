import envs.stir.stir_utils as stir_utils
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(9, 9))

ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

x = np.linspace(0, 1, 1000)
y1 = stir_utils.get_negative_exp(x)
y2 = stir_utils.get_subtract_negative_exp_from_one(x)
y3 = stir_utils.get_sigmoid(x)
y4 = stir_utils.get_inverted_sigmoid(x)

c1, c2, c3, c4 = "black", "black", "black", "black"
l1, l2, l3, l4 = "exp(-x)", "1-exp(x)", "1/(1+exp(-x))", "1/(1+exp(x))"

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)

ax1.set_title("a", y=-0.15, fontsize=42)
ax2.set_title("b", y=-0.15, fontsize=42)
ax3.set_title("c", y=-0.15, fontsize=42)
ax4.set_title("d", y=-0.15, fontsize=42)

ax1.plot(x, y1, color=c1, label=l1)
ax2.plot(x, y2, color=c2, label=l2)
ax3.plot(x, y3, color=c3, label=l3)
ax4.plot(x, y4, color=c4, label=l4)
ax1.legend(fontsize=24, loc="upper right")
ax2.legend(fontsize=24, loc="upper right")
ax3.legend(fontsize=24, loc="upper right")
ax4.legend(fontsize=24, loc="upper right")
fig.tight_layout()
plt.show()
# fig.savefig("~/Downloads/sample.svg")
