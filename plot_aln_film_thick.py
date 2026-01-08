from __future__ import annotations

import numpy as np


def main() -> None:
    data = np.genfromtxt(
        "comsol_data/aln_film_thick.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )

    h_over_lambda = data["h_over_lambda"]
    v_open = data["v_open"]
    v_short = data["v_short"]
    k2 = data["K2"]

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    plt.rcParams["font.family"] = "Meiryo"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    ax1.plot(h_over_lambda, v_open, marker="o", label="v_open")
    ax1.plot(h_over_lambda, v_short, marker="s", label="v_short")
    ax1.set_ylabel("Velocity (m/s)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="best")
    ax1.set_title("AlN film thickness sweep")

    ax2.plot(h_over_lambda, k2 * 100.0, color="tab:green", marker="^")
    ax2.set_xlabel(r"$t_{AlN}/\lambda$")
    ax2.set_ylabel("$K^2$ (%)")
    ax2.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
