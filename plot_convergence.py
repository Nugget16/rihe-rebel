import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

df = pd.read_csv("sweep_results.csv")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(df["iters"], df["vs_random"], marker="o", label="MCCFR vs Random")
ax.plot(df["iters"], df["vs_heuristic"], marker="s", label="MCCFR vs Heuristic")
ax.axhline(y=2.018, color="blue", linestyle="--", alpha=0.5, label="GSI vs Random")
ax.axhline(y=-1.190, color="orange", linestyle="--", alpha=0.5, label="GSI vs Heuristic")
ax.set_xlabel("Training Iterations")
ax.set_ylabel("Seat-averaged EV (chips/hand)")
ax.set_title("MCCFR Convergence")
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: "0" if x == 0 else (f"{int(x/1000)}k" if x < 1_000_000 else "1M")))
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("mccfr_convergence.pdf", dpi=300)
plt.savefig("mccfr_convergence.png", dpi=300)
print("Saved.")