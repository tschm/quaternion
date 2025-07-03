# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo==0.14.9",
#     "pandas==2.3.0",
#     "numpy==2.3.1",
#     "matplotlib==3.10.0",
# ]
# ///

import marimo

__generated_with = "0.14.10"
app = marimo.App()

with app.setup:
    import marimo as mo
    import numpy as np
    import pandas as pd
    from numpy.linalg import eig


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
        # Rotationen in 3D



        """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
        ## Rotationen in 3D
        * Die Rotationsmatrizen $\textrm{SO}(3)$
        * Die Eulerachse
        * Die Quaternionen
        """
    )
    return


@app.function
def pp(ax, point, **kwargs):
    ax.plot([0, point[0]], [0, point[1]], [0, point[2]], **kwargs)


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
        ## Rotationsmatrizen $\textrm{SO}(3)$

        $3 \times 3$ Matrizen, orthonormale Spalten und $\det(A) = 1$.

        Fertig? Matrizen manchmal etwas unhandlich. $9$ Einträge, Fragen der Stabilität?

        """
    )
    return


@app.function
def rand_so3():
    # We construct a random element in SO(3)
    A = np.random.randn(3, 3)
    # normalize the first column
    A[:, 0] = A[:, 0] / np.linalg.norm(A[:, 0], 2)
    # make the 2nd column orthogonal to first column
    A[:, 1] = A[:, 1] - np.dot(A[:, 0], A[:, 1]) * A[:, 0]
    # normalize the second column
    A[:, 1] = A[:, 1] / np.linalg.norm(A[:, 1], 2)
    # The third column is just the cross product of the first two columns => det = 1
    A[:, 2] = np.cross(A[:, 0], A[:, 1])

    print("Determinante von A: {}".format(np.linalg.det(A)))
    print("Check if columns are orthonormal")
    print(np.linalg.norm(np.dot(A.T, A) - np.eye(3), "fro"))
    return A


@app.cell
def _():
    for i in range(0, 3):
        values, vectors = eig(rand_so3())
        d = dict()
        for j, value in enumerate(np.sort(values)):
            d[j] = {
                "Real": value.real,
                "Imag": value.imag,
                "Abs": np.abs(value),
                "phi": np.angle(value, deg=True),
            }

        print(
            "Eigenwerte von A:\n{}\n".format(
                pd.DataFrame(d).transpose()[["Real", "Imag", "Abs", "phi"]]
            )
        )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
        * Es gibt immer einen Eigenwert 1
        * Eine reelle 3 x 3 Matrix hat immer mind. einen reellen Eigenwert
        * Das charakteristische Polynom einer 3x3 Matrix hat den Grad 3.
        * In Polynom vom Grad 3 hat mind eine reelle Nullstelle (das ist nicht wahr fuer Polynome vom Grad 2,4,6,...)
        * Was können Sie über die beiden anderen Eigenwerte sagen?
        * Was können Sie über das Product der Eigenwerte sagen?
        * Was bedeutet es geometrisch so einen Eigenwert zu haben?
        """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
        ## Beobachtungen
        * Es existiert in Vector x (reell), so dass A*x = x. Diese Vector beschreibt die __Euler Achse__
        * Die beiden anderen Eigenwerte sind $z$ und $\bar{z}$.
        * $A*x=x$ ist auch die Gleichung für die stationäre Verteilung einer Markovkette. $A$ ist dann die Transition Matrix.
        """
    )
    return


# @app.cell
# def _():
#    Image("Euler_AxisAngle.png")
#    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
        Jede Rotation in 3d ist also eine 2d Drehung um die Euler Achse.
        Eine solche Achse existiert immer!

        Punkte auf der Euler Achse bleiben unverändert.
        Jeder Punkt liegt in einer Ebene orthogonal zur Euler Achse und bleibt auch nach der Rotation in dieser Ebene.

        Die Berechnung der Euler Achse und des Drehwinkels $\varphi$ ist etwas technisch (ergo Hausaufgabe).
        """
    )
    return


@app.cell
def _():
    # There are always 3 of them!
    # one always seems to be 1
    # The other two eigenvalues are $z$ and $\bar{z}$
    # Can we compute $\mathbf{A}*n=n$ without too much hassle?
    # What's the interpretation of n?

    # A^T*n = A^T*A*n=n
    # (A^T - A)*n = 0
    # B = A^T-A = 0
    return


@app.cell
def _():
    A_1 = rand_so3()

    def euler(A, eps=1e-10):
        assert np.abs(np.linalg.det(A) - 1) < eps, (
            "Die Determinante der Matrix A ist {}".format(np.linalg.det(A))
        )
        assert np.linalg.norm(np.dot(A.T, A) - np.eye(3), "fro") < eps, (
            "Die Matrix A ist nicht orthonormal"
        )
        assert A.shape == (3, 3), "Die Matrix A ist nicht 3 x 3"
        theta = np.arccos((np.trace(A) - 1) / 2)
        e_1 = A[2, 1] - A[1, 2]
        e_2 = A[0, 2] - A[2, 0]
        e_3 = A[1, 0] - A[0, 1]
        n = np.array([e_1, e_2, e_3])
        assert np.abs(np.linalg.norm(n, 2) - 2 * np.sin(theta)) < eps
        return (n / np.linalg.norm(n, 2), theta)

    axis, angle = euler(A_1)
    print(np.linalg.norm(np.dot(A_1, axis) - axis, 2))
    print(180 * angle / np.pi)
    return (euler,)


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
        * Die Idee ist es die Rotation also Paar aus Achse $(x_1, x_2, x_3)$ mit $x_1^2 + x_2^2 + x_3^2=1$ und Winkel $\varphi$ anzugeben. Das schaffen wir eben genau mit den Quaternionen uns insbesondere den Versoren:
        \\[
        q = \cos(\varphi/2) + \sin(\varphi/2) (ix_1 + jx_2 + kx_3)
        \\]
        * In Vector $x \in \mathbb{R}^3$ kann mit der reinen Quaternion
        \\[
        x_r = ix_1 + jx_2 + kx_3
        \\]
        identifiziert werden.
        * Die Multiplication der Quaternionen folgt
        \\[
        i^2 = j^2 = k^2 = ijk = -1
        \\]
        * Die Rotation ist dann der vektorwertige Teil der Quaternion
        \\[
        x' = q x_r \bar{q}
        \\]

        """
    )
    return


@app.cell
def _(euler, np):
    def fromSO3(A):
        axis, angle = euler(A)
        vector = np.sin(angle / 2) * axis
        return Quaternion(
            np.array([np.cos(angle / 2), vector[0], vector[1], vector[2]])
        )

    def fromSO3_fast(A):
        q_r = 0.5 * np.sqrt(1 + np.trace(A))
        q_i = (A[2, 1] - A[1, 2]) / (4 * q_r)
        q_j = (A[0, 2] - A[2, 0]) / (4 * q_r)
        q_k = (A[1, 0] - A[0, 1]) / (4 * q_r)
        return Quaternion(np.array([q_r, q_i, q_j, q_k]))

    class Quaternion:
        def __init__(self, q):
            assert len(q) == 4
            self.__q = q
            # make the array immutable
            self.__q.flags.writeable = False

        @property
        def conjugate(self):
            return Quaternion(
                np.array([self.__q[0], -self.__q[1], -self.__q[2], -self.__q[3]])
            )

        @property
        def versor(self):
            return Quaternion(self.__q / self.norm)

        @property
        def norm(self):
            return np.linalg.norm(self.__q, 2)

        def __repr__(self):
            return "Q{}".format(self.__q)

        def __mul__(self, other):
            if isinstance(other, Quaternion):
                w1, x1, y1, z1 = tuple(self.__q)
                w2, x2, y2, z2 = tuple(other.__q)

                w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
                x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
                y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
                z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
                return Quaternion(np.array([w, x, y, z]))
            else:
                raise NotImplementedError()

        def __rmul__(self, other):
            return Quaternion(other * self.__q)

        def __add__(self, other):
            return Quaternion(self.__q + other.__q)

        def rotate(self, x):
            # construct the pure quaternion
            z = Quaternion(np.array([0, x[0], x[1], x[2]]))
            assert np.abs(self.norm - 1) < 1e-10, "The quaternion has to be a versor!"
            return (self * z * self.conjugate).vector

        @property
        def so3(self):
            # there are less expensive formulas for this step. Howe
            return np.apply_along_axis(self.rotate, 0, np.eye(3))

        @property
        def vector(self):
            return self.__q[1:]

        @property
        def real(self):
            return self.__q[0]

    return fromSO3, fromSO3_fast


@app.cell
def _():
    A_2 = rand_so3()
    print(A_2)
    print(np.linalg.det(A_2))
    print(np.linalg.norm(np.dot(A_2.T, A_2) - np.eye(3), "fro"))
    return (A_2,)


@app.cell
def _(A_2, fromSO3, fromSO3_fast, np):
    q1 = fromSO3(A_2)
    q2 = fromSO3_fast(A_2)
    print(q1)
    print((q1 + -1 * q2).norm)
    print(np.linalg.norm(A_2 - q2.so3))
    return q1, q2


@app.cell
def _(q1):
    print("Quaternion {}".format(q1))
    print("Norm       {}".format(q1.norm))
    print("Real       {}".format(q1.real))
    print("Vector     {}".format(q1.vector))
    print("Versor     {}".format(q1.versor))
    print("Conjugate  {}".format(q1.conjugate))
    return


@app.cell
def _(q2):
    print("Quaternion Vector (Euler-Achse) {}".format(q2.vector))
    print("Und rotiert                     {}".format(q2.rotate(x=q2.vector)))
    return


@app.cell
def _(A_2, q2):
    print("Die Einheitsvektoren rotiert  \n{}".format(q2.so3))
    print("Die Rotationsmatrix           \n{}".format(A_2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Aber warum?
        * Die Rotation von (vielen) Punkten mittels Quaternionen ist verglichen mit Rotationsmatrizen teuer.
        * Interessanter für Rotationen von Rotationen, numerisch stabiler...
        * Interpolation von Quaternionen (SLERP)...
        """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
        ## Alternative Representations of SO(3)
        siehe https://github.com/moble/quaternion

        Euler angles are pretty much the worst things ever and it makes me feel bad even supporting them. Quaternions are faster, more accurate, basically free of singularities, more intuitive, and generally easier to understand. You can work entirely without Euler angles (I certainly do). You absolutely never need them.

        Es gibt viele mögliche Representations der Element von SO(3). Siehe https://en.wikipedia.org/wiki/Charts_on_SO(3)


        """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
        Sie fliegen von Biel direkt zum Nordpol. Was ist Ihr Längengrad? Sie überfliegen den Nordpol und nehmen geradeaus Kurs auf Hawaii... Sie zeichnen den Längengrad auf. Was passiert am Nordpol?
        """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
        ### SLERP (interpolation of two quarterions)

        * special case: $q_0 = 1$, $q_1 = q = \cos(\theta) + \sin(\theta)\mathbf{v}$ (Polarform)

        \\[
        SLERP(1, q, t) = q^t = \cos(t\theta) + \sin(t\theta)\mathbf{v}
        \\]

        * normal case: $q_0$ and $q_1$ both unit quaternions:

        \\[
        SLERP(q_0, q_1, t) = (q_1 q_0^{-1})^t q_0
        \\]



        """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
        ## Zusammenfassung 1:
        * Komplexe Zahlen eng verwandt mit den Rotationsmatrizen SO(2).
        * Quaternionen eng verwandt mit den Rotationsmatrizen SO(3).
        * SO(2) eng verwandt mit SO(3), also Komplexe Zahlen eng verwandt mit Quaternionen.
        """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
        ## Zusammenfassung 2:
        * Eine Rotation in 3D ist immer eine Rotation um eine feste Euler-Achse.
        * Die Euler-Achse ist der Eigenvektor (zum Eigenwert 1.0) einer Matrix aus SO(3).
        * Die Euler-Achse beschreibt den vektorwertigen Anteil der Quaternion.
        """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
        ## Zusammenfassung 3:
        * Quarternionen insbesondere hilfreich bei vielen Rotationen weniger Punkte.
        * Rotationsmatrizen hilfreich bei wenigen Rotationen vieler Punkte.
        * Glückliches Leben auch ohne Quaternionen möglich, unmöglich aber ohne Lineare Algebra und Komplexe Zahlen.
        """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
        ## Hausaufgaben:
        * Wie weit ist es von Biel nach Sydney
        * Sie leben in Biel. Sie hätten gerne mehr Sonne und weniger Nebel. Drehen Sie deshalb Biel nach Nizza. Hinweis: Drehen Sie um die Achse, die senkrecht auf der Ebene Biel-Nizza-Erdmittelpunkt steht.
        * Erweitern Sie die Python Klasse um eine Method SLERP.
        * Wer war Olinde Rodrigues? Die Method so3 in der Klasse Quaternion ist nicht wirklich optimal. Schlagen Sie eine Alternative vor...
        * Beweisen Sie, dass alle Eigenwerte einer orthonormalen Matrix Betrag $1$ haben.
        * Beweisen Sie, dass jede Matrix aus SO(5) eine Euler-Achse hat.
        * Sei $p$ in Polynom mit ungeradem Grad. Beweisen Sie, dass es mind. eine reelle Nullstelle hat.

        """
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
