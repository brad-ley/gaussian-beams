from __future__ import annotations

from pathlib import Path as P

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

if __name__ == "__main__":
    plt.style.use(["science"])
    rc("text.latex", preamble=r"\usepackage{cmbright}")
    rcParams = [
        ["font.family", "sans-serif"],
        ["font.size", 14],
        ["axes.linewidth", 1],
        ["lines.linewidth", 2],
        ["xtick.major.size", 5],
        ["xtick.major.width", 1],
        ["xtick.minor.size", 2],
        ["xtick.minor.width", 1],
        ["ytick.major.size", 5],
        ["ytick.major.width", 1],
        ["ytick.minor.size", 2],
        ["ytick.minor.width", 1],
    ]
    plt.rcParams.update(dict(rcParams))

LJ1 = 25
LJ2 = 15


class Lens:
    def __init__(self, f: float, pos: float = -1, r: float = 25.4) -> Lens:
        """__init__.

        :param f: focal length of lens (mm)
        :param pos: distance from last optical element (mm) [default is 2x focal length]
        :param r: radius of lens (mm) [default 25.4 mm]
        """
        self.f = f
        self.r = r
        if pos == -1:
            self.pos: float = 2 * f
        else:
            self.pos: float = pos

        self.abcd = np.array([[1, 0], [-1 / self.f, 1]])
        if f < 0 or pos < 0 or r < 0:
            raise Exception("All optical elements require non-negative inputs.")

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__:<9} @ {float(self.pos):>4.1f} mm".ljust(LJ1, ".")
            + f"f={self.f:.1f} mm".ljust(LJ2)
            + f"r={self.r:.1f} mm".ljust(LJ2)
        )


class Mirror(Lens):
    def __init__(self, f: float, pos: float = -1, r: float = 25.4):
        """__init__.

        :param f: focal length of mirror (mm)
        :param pos: distance from last optical element (mm) [default is 2x focal length]
        :param r: radius of lens (mm) [default 25.4 mm]
        """
        super().__init__(f, pos, r)


class Aperture(Lens):
    def __init__(self, pos: float, r: float = 25.4):
        """__init__.

        :param pos: distance from last optical element
        :param r: radius of aperture (mm) [default 25.4 mm]
        """
        self.r = r
        super().__init__(np.inf, pos, r)

    def __repr__(self):
        return (
            f"{type(self).__name__:<9} @ {float(self.pos):>4.1f} mm".ljust(LJ1, ".")
            + "".ljust(LJ2, ".")
            + f"r={self.r:.1f} mm".ljust(LJ2)
        )


class FreeSpace(Lens):
    def __init__(self, z, n=1):
        """__init__.

        :param z: distance to propagate
        :param n: index of refraction of medium [default 1]
        """
        self.z = z
        self.n = n
        self.abcd = np.array([[1, self.z / self.n], [0, 1]])

    def __repr__(self):
        return f"{type(self).__name__:<9} for {self.z:>4} mm".ljust(
            LJ1, "."
        ) + f"n={self.n}".ljust(LJ2)


class Beam:
    def q(self, R, lam, w):
        return (1 / R - 1j * lam / (np.pi * w**2)) ** -1  # type: ignore

    def __init__(self, lam, w, R=np.inf):
        """__init__.

        :param lam: wavelength (mm)
        :param w: beam radius (mm)
        :param R: initial radius of curvature [default at waist, therefore R=infinite]
        :param z_axis: z-axis to simulate along [default total length of optical system plus focal length of last element]
        """

        self.lam = lam
        self.w = w
        self.R = R

    def simulate(
        self,
        elements: list[Lens] | list[Mirror] | list[Aperture] = None,
        fiber_integral_radius: float = -1,
        z_axis: tuple = (),
    ) -> Lens:
        """simulate.

        :param elements: list of optical elements
        :param fiber_integral_radius: list of optical elements
        :param z_axis: np array for z axis (mm)

        Returns
            self
        """
        if any(z_axis):
            self.z_axis = np.array(z_axis)
        if not hasattr(self, "z_axis"):
            add = 0
            if elements[-1].f < np.inf:
                add = elements[-1].f
            self.z_axis = np.linspace(
                0,
                np.sum([optic.pos for optic in elements]) + add,
                1000,
            )

        self._q = np.zeros(self.z_axis.shape).astype("complex128")
        self._q[0] = self.q(self.R, self.lam, self.w)
        self._amp = 1

        if elements is None:
            self.optics = []
        else:
            self.optics = [ii for ii in elements if type(ii) in {Lens, Mirror, Aperture}]

        optic_positions = []
        for optic in elements:
            z = 0
            if optic_positions:
                z = optic_positions[-1]
            optic_positions.append(optic.pos + z)

        self.elements = elements[:].copy()  # pass by value
        self.optic_positions = optic_positions.copy()  # pass by value

        optic_count = 0
        h = False
        for i, z in enumerate(self.z_axis):
            if i > 0:  # make sure we aren't at first index
                if optic_positions and z > optic_positions[0]:
                    mat = self.optics[optic_count].abcd
                    aperture = self.optics[optic_count].r
                    optic = optic_positions.pop(0)
                    optic_count += 1
                else:
                    mat = FreeSpace(
                        z=z - self.z_axis[i - 1],
                    ).abcd  # no optics left so just propagate through to end of z_axis
                    aperture = np.inf

                self._q[i] = (mat[0][0] * self._q[i - 1] + mat[0][1]) / (
                    mat[1][0] * self._q[i - 1] + mat[1][1]
                )
                omeg = np.sqrt(-self.lam / np.pi / np.imag(1 / self._q[i]))
                self._amp *= 1 - np.exp(-((aperture / omeg) ** 2))

                if omeg > aperture:
                    self._q[i] = 1 / (
                        np.real(1 / self._q[i]) - 1j * self.lam / (np.pi * aperture**2)
                    )

        self.w = np.sqrt(-self.lam / np.pi / np.imag(1 / self._q))

        if fiber_integral_radius == -1:
            fiber_integral_radius = self.w[0]
        self.fiber_integral = np.exp(
            -((self.w[-1] - fiber_integral_radius) ** 2) / (fiber_integral_radius) ** 2,
        )
        print(
            f"Converted power = {self._amp:.2f} ({10 * np.log10(self._amp):.1f} dB loss)\n"
            f"Fiber coupling = {self.fiber_integral:.2f}",
        )
        return self

    def plot(
        self,
        encircled_energy: float = 100 * (1 - np.exp(-2)),
        full: str = "enabled",
        savepath: str = "",
    ) -> None:
        """
        plot.

        :param encircled_energy: % encircled energy radius to show in plot [default beam waist w]
        :param full: True -- plot only positive r, False -- plot r and -r [default True]
        :param savepath: folder path to save figure to [default no path, file not saved]
        """

        def legend_without_duplicate_labels(ax: all) -> None:
            handles, labels = ax.get_legend_handles_labels()
            unique = [
                (h, ll) for i, (h, ll) in enumerate(zip(handles, labels)) if ll not in labels[:i]
            ]
            ax.legend(*zip(*unique), frameon=True)

        f, a = plt.subplots(
            figsize=(7, 5),
            num=rf"Beam waist and {encircled_energy:.1f}%"
            f" encircled ({10 * np.log10(self._amp):.1f} dB loss)",
        )
        w = self.w[:]
        z_axis = self.z_axis[:]
        for position in self.optic_positions[::-1]:  # reversed to not move indices after inserting
            if len(np.where(self.z_axis > position)[0]) > 0:
                w = np.insert(w, np.where(self.z_axis > position)[0][0], np.nan)
                z_axis = np.insert(z_axis, np.where(self.z_axis > position)[0][0], np.nan)

        for i, position in enumerate(self.optic_positions):
            if (
                type(self.elements[i]) is Aperture
                and self.optic_positions[i] <= np.floor(self.z_axis.max() * 10) / 10
            ):
                a.vlines(
                    position,
                    ymin=self.elements[i].r,
                    ymax=max(self.w.max(), self.elements[i].r + 5),
                    color="r",
                    label=type(self.elements[i]).__name__,
                )
                if full == "enabled":
                    a.vlines(
                        position,
                        ymin=min(-self.w.max(), -self.elements[i].r - 5),
                        ymax=-self.elements[i].r,
                        color="r",
                        label=type(self.elements[i]).__name__,
                    )
            elif type(self.elements[i]) in {Lens, Mirror}:
                ls = ":" if type(self.elements[i]) is Lens else "--"
                if full == "enabled":
                    a.plot(
                        [position, position],
                        [self.optics[i].r, -self.optics[i].r],
                        c="k",
                        alpha=0.5,
                        ls=ls,
                        label=type(self.elements[i]).__name__,
                    )
                else:
                    a.plot(
                        [position, position],
                        [self.optics[i].r, 0],
                        c="k",
                        alpha=0.5,
                        ls=ls,
                        label=type(self.elements[i]).__name__,
                    )

                label = r"$\infty$" if self.optics[i].f == np.inf else f"{self.optics[i].f:d}"
                a.annotate(
                    # rf"${self.optics[i].f:d}\,$mm",
                    label,
                    (position, self.optics[i].r * 1.05),
                    horizontalalignment="center",
                )

        scale = np.sqrt(-1 / 2 * np.log((100 - encircled_energy) / 100))

        a.plot(z_axis, w * scale, c="k")
        if full == "enabled":
            bottom = -self.w
            a.plot(z_axis, -w * scale, c="k")
        else:
            bottom = 0
        a.fill_between(
            self.z_axis,
            y1=bottom,
            y2=self.w,
            facecolor="k",
            alpha=0.25,
            # label=r"$\omega(z)$",
        )

        a.set_ylabel("Radial distance (mm)")
        a.set_xlabel("$z$-distance (mm)")
        a.set_xlim([self.z_axis.min(), self.z_axis.max() * 1.025])
        a.set_ylim(top=a.get_ylim()[1] * 1.1)
        legend_without_duplicate_labels(a)
        savename = ""
        for optic in self.elements:
            savename += f"{type(optic).__name__[0]}{
                int(optic.f) if optic.f != np.inf else optic.f
            }@{int(optic.pos)}_"
        savename = savename.rstrip("_")
        savename += ".png"
        a.set_title(
            rf"$\omega(z)$ and {encircled_energy:.1f}\%"
            f" encir. ({10 * np.log10(self._amp):.1f} dB loss)",
        )
        f.savefig(P(savepath).joinpath(savename), dpi=600)
        f.show()


def main() -> None:
    elements = [
        Mirror(f=250, pos=250, r=50),
        Mirror(f=250, pos=500, r=50),
        Aperture(pos=250 + 0 + 89 - 89 / np.sqrt(2), r=12.7),
        Mirror(f=89, pos=89 / np.sqrt(2)),
        Mirror(f=np.inf, pos=89),
        Mirror(f=89, pos=89),
        Aperture(pos=89 / np.sqrt(2), r=12.7),
        Mirror(f=np.inf, pos=89 - 89 / np.sqrt(2), r=30),
    ]
    print(*elements, sep="\n")

    B = Beam(lam=1.25, w=5.7)
    B.simulate(elements=elements, fiber_integral_radius=5).plot(
        encircled_energy=95,
        savepath="/Users/Brad/Desktop/",
    )


if __name__ == "__main__":
    main()
    plt.show()
