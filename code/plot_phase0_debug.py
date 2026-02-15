import numpy as np
import matplotlib.pyplot as plt
import sys
import os


def load_debug(path):
    """Load Phase0 debug npy file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    recs = np.load(path, allow_pickle=True)
    return recs


def main(path):
    recs = load_debug(path)

    if len(recs) == 0:
        print("No records found.")
        return

    # Convert list[dict] -> dict of arrays
    keys = recs[0].keys()
    data = {k: np.array([r[k] for r in recs]) for k in keys}

    t = data["time_idx"]

    # ---------------- Figure 1: ESS ----------------
    plt.figure()
    plt.plot(t, data["ess"])
    plt.title("ESS over time")
    plt.xlabel("Time index")
    plt.ylabel("ESS")
    plt.grid(True)

    # ---------------- Figure 2: Scan score range ----------------
    plt.figure()
    plt.plot(t, data["scan_min"], label="scan_min")
    plt.plot(t, data["scan_med"], label="scan_med")
    plt.plot(t, data["scan_max"], label="scan_max")
    plt.title("Scan score range over particles")
    plt.xlabel("Time index")
    plt.ylabel("Scan score")
    plt.legend()
    plt.grid(True)

    # ---------------- Figure 3: Distribution fractions ----------------
    plt.figure()
    plt.plot(t, data["mean_frac_hit"], label="hit")
    plt.plot(t, data["mean_frac_short"], label="short")
    plt.plot(t, data["mean_frac_max"], label="max")
    plt.plot(t, data["mean_frac_rand"], label="rand")
    plt.title("Mean mixture fractions per beam (best particle)")
    plt.xlabel("Time index")
    plt.ylabel("Mean fraction")
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_phase0_debug.py results/debug_phase0.npy")
    else:
        main(sys.argv[1])
