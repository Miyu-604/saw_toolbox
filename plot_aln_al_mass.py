from __future__ import annotations

import numpy as np


def main() -> None:
    data = np.genfromtxt(
        "comsol_data/aln_al_mass.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )

    t_over_lambda = data["t_over_lambda"]
    v_metal = data["v_metal"]
    v_idt_0p5 = data["v_idt_0p5"]
    v_idt_0p7 = data["v_idt_0p7"]

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    plt.rcParams["font.family"] = "Meiryo"

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(t_over_lambda, v_metal, marker="o", label="metalized")
    ax.plot(t_over_lambda, v_idt_0p5, marker="s", label="IDT w = 0.5")
    ax.plot(t_over_lambda, v_idt_0p7, marker="^", label="IDT w = 0.7")
    ax.set_xlabel(r"$t_{Al}/\lambda$")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("AlN : Al thickness sweep")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
