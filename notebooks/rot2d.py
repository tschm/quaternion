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

__generated_with = "0.14.9"
app = marimo.App(layout_file="layouts/rot2d.slides.json")

with app.setup:
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy.linalg import eig
    from numpy.random import rand


@app.cell(hide_code=True)
def _():
    mo.md(r"""# Rotationen in 2d""")
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## Warnung

    Quaternions are the things that scare all manner of mice and men.  They are the things that go bump in the night.  They are the reason your math teacher gave you an F.  They are all that you have come to fear, and more.  **Quaternions are your worst nightmare**.

    http://www.cprogramming.com/tutorial/3d/quaternions.html
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## Rotationen in 2D
    * Wie rotieren wir z.B. in Polygon?
    * Wann ist eine lineare Abbildung eine Rotation?
    * Rotationsmatrizen und komplexe Zahlen.
    * Hoffnung für 3D?
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## Unser Ziel:
    * Wer interessiert sich warum für Quaternionen? Sollten wir das auch tun?
    * Die mathematische Heimat und Verwandten der Quaternionen.
    * Erste kleinere Anwendungen...
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Welches Spiel war das brutalste Game aller Zeiten?""")
    return


@app.function(hide_code=True)
def pp(ax, point, **kwargs):
    ax.plot([0, point[0]], [0, point[1]], **kwargs)


@app.function
def baseimage(a1, a2):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, aspect="equal")
    pp(ax, point=np.array([1, 0]), color="b", linewidth=5)
    pp(ax, point=np.array([0, 1]), color="r", linewidth=5)
    pp(ax, point=np.array(a1), color="b", linewidth=5, linestyle="--")
    pp(ax, point=np.array(a2), color="r", linewidth=5, linestyle="--")
    plt.grid()
    plt.show()


@app.cell
def _():
    _phi = np.pi / 6
    baseimage(a1=[np.cos(_phi), np.sin(_phi)], a2=[-np.sin(_phi), np.cos(_phi)])
    return


@app.cell
def _():
    _phi = np.pi / 6
    baseimage(a1=[np.cos(_phi), np.sin(_phi)], a2=[-2 * np.sin(_phi), 2 * np.cos(_phi)])
    return


@app.cell
def _():
    _phi = np.pi / 6
    baseimage(a1=[np.cos(_phi), np.sin(_phi)], a2=[np.sin(_phi), np.cos(_phi)])
    return


@app.cell
def _():
    _phi = np.pi / 6
    baseimage(a1=[np.cos(_phi), np.sin(_phi)], a2=[np.sin(_phi), -np.cos(_phi)])
    return


@app.cell
def _():
    _phi = np.pi / 6
    baseimage(a1=[-1, 0], a2=[0, -1])
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ### Die Gruppe $\textrm{SO}(2)$

    * Spalten von A müssen orthonormal sein und $\det(\mathbf{A}) = 1$.

    * Teste mit $\lVert\mathbf{A}^{T}\mathbf{A} - \mathbf{I}\rVert < \varepsilon$ und expliziter Berechnung von $\det(\mathbf{A})$.

    * Diese Matrizen bilden eine Gruppe (bzgl. der Multiplication). $\mathbf{A}_1, \mathbf{A}_2 \in \textrm{SO}(2)$ dann gilt $\mathbf{A}_1 \mathbf{A}_2 \in \textrm{SO}(2)$.

    ### Die Rotationen in 2D sind die Elemente von $\textrm{SO}(2)$.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ### Komplexe (Einheits-)Zahlen also Rotation
    Wir können aber auch jeden Punkt $(a,b) \in \mathbb{R}^2$ mit der komplexen Zahl $z=a+i b$ eindeutig identifizieren.
    Die Drehung um den Winkel $\varphi$ entspricht dann einfach einer komplexen Multiplication
    \\[
    u z = z u
    \\]
    wobei gilt $u=\cos(\varphi) + i\sin(\varphi)$. Zurück ins Reelle per Real- und Imaginärteil.

    $\textrm{SO}(2)$ eng verwandt mit den komplexen Zahlen auf der Einheitssphäre (der Rand der Einheitsscheibe).

    ### Komplexe Zahlen einfacher also Rotationsmatrizen, aber Schritt in höhere Dimensionen unklar.
    """
    )
    return


@app.cell
def _():
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, aspect="equal")
    pp(ax, point=np.array([1, 2]), color="b", linewidth=5)
    pp(ax, point=np.array([0, 1]), color="r", linewidth=5)
    pp(ax, point=np.array([-2, 1]), color="b", linewidth=5, linestyle="--")
    plt.grid()
    plt.show()
    # z=1+2i
    # u=i
    # u*z=i+i*2i=-2+i  weil i*i=-1
    # Rotation um pi/2
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ### Hoffnung für 3D?
    * Rotationen in 2D beschrieben durch $2$-dimensionale Zahlen.
    * Komplexe Zahlen aber nutzlos in höheren Dimensionen. Gibt es $3$ dimensionale Zahlen? Gibt es eine Verallgemeinerung der komplexen Zahlen für höhere Dimension.
    * Gibt es $\textrm{SO}(3)$ verwandt mit $X$? Was ist $X$?
    * Rotationen in 2D beschrieben mit nur einem Parameter. Wieviele freie Parameter brauchen wir für 3D?
    * Matrizen verfügbar in allen Dimensionen (ganzzahlig), aber etwas unhandlich.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ### Animationen in 2D?
    * Wir drehen Punkte $z = x +i y$ um den Winkel $\varphi$.
    * Wir interpolieren diese Drehung mittels $\varphi(t) = \frac{t}{T}\phi$.
    * Die Animiation ist dann einfach
    
    \[
    z(t) = (\cos(\varphi(t)) + i\sin(\varphi(t))) z(0)
    \]
    
    * Eine Animation ist hier nur eine Hintereinanderausführung vieler, vieler Rotationen.
    * Gleichmässige Winkelgeschwindigkeit impliziert weniger **Ruckeln**.
    """
    )
    return


@app.function
def so2(_phi):
    return np.array([[np.cos(_phi), -np.sin(_phi)], [np.sin(_phi), np.cos(_phi)]])


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    Kurzes Quiz
    
    * Eine Drehung in 2D stellen wir also Rotationsmatrix oder komplexe (Einheits)Zahl dar.
    * Was gilt für die Spalten einer solchen Rotationsmatrix?
    * Was gilt für die Determinante einer solchen Matrix?
    * Wie testet man die Orthonormalität einer Matrix?
    * Schreiben Sie eine Klasse, die eine Rotation in 2D beschreibt.
    * Multiplizieren Sie 10000 zufällig gewählte Rotationsmatrizen und untersuchen Sie, ob das Product auch wirklich eine Rotationsmatrix ist.

    Der Zugang via Matrizen öffnet den Weg in alle höheren Dimensionen, $\textrm{SO}(3)$, $\textrm{SO}(4)$, ...
    Der Zugang via komplexer Zahlen erleichtert das Verständnis für Quaterionen.

    Die Rotationsmatrizen sind letztlich durch nur einen(!) Parameter $\varphi$ bestimmt.
    """
    )
    return


@app.cell
def _():
    _A = so2(np.pi / 6)
    print(_A)
    print("Eigenvalues: ")
    print(eig(_A)[0])
    print("Eigenvectors (columns): ")
    print(eig(_A)[1])
    return


@app.cell
def _():
    _A = np.eye(2)
    for r in rand(10000):
        _A = np.dot(so2(r), _A)
    print(_A)
    print(np.dot(_A.T, _A) - np.eye(2))
    print(np.linalg.det(_A))
    return


if __name__ == "__main__":
    app.run()
