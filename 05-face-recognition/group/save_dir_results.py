import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt

from cvproj_exc.classifier import NearestNeighborClassifier
from cvproj_exc.config import Config
from cvproj_exc.evaluation import OpenSetEvaluation

false_alarm_rate_range = np.logspace(-3.0, 0, 1000, endpoint=False)

classifier = NearestNeighborClassifier()
evaluation = OpenSetEvaluation(
    classifier=classifier,
    false_alarm_rate_range=false_alarm_rate_range
)

evaluation.prepare_input_data(Config.EVAL_TRAIN_DATA, Config.EVAL_TEST_DATA)
results = evaluation.run()

fars = false_alarm_rate_range
irs = results["identification_rates"]
ths = results["similarity_thresholds"]

idx_1 = np.argmin(np.abs(fars - 0.01))
idx_10 = np.argmin(np.abs(fars - 0.10))

with open("results/logs/dir_metrics.txt", "w") as f:
    f.write("===== DIR Evaluation Results =====\n")
    f.write(f"DIR@FAR=1%      : {irs[idx_1]:.6f}\n")
    f.write(f"Threshold@1%    : {ths[idx_1]:.6f}\n")
    f.write(f"DIR@FAR=10%     : {irs[idx_10]:.6f}\n")
    f.write(f"Threshold@10%   : {ths[idx_10]:.6f}\n")

    valid_90 = np.where(irs >= 0.90)[0]
    if len(valid_90) > 0:
        best_idx = valid_90[0]
        f.write(f"Minimum FAR with DIR >= 90%: {fars[best_idx]:.6f}\n")
        f.write(f"Threshold there           : {ths[best_idx]:.6f}\n")
    else:
        f.write("DIR never reaches 90%.\n")

plt.figure()
plt.semilogx(fars, irs, linewidth=3, linestyle="--")
plt.grid(True)
plt.axis([fars[0], fars[-1], 0, 1])
plt.xlabel("False alarm rate")
plt.ylabel("Identification rate")
plt.savefig("results/plots/dir_curve.png", dpi=300, bbox_inches="tight")

np.savez(
    "results/plots/dir_curve_values.npz",
    false_alarm_rates=fars,
    identification_rates=irs,
    similarity_thresholds=ths,
)

print(open("results/logs/dir_metrics.txt").read())
print("Saved: results/plots/dir_curve.png")
print("Saved: results/plots/dir_curve_values.npz")
