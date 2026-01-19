Matplotlib Style
================

brypy includes a matplotlib style file designed for publications in American
Astronomical Society (AAS) journals (ApJ, ApJL, ApJS, AJ, etc.).

Features
--------

- **Fonts**: Serif fonts (Times New Roman) at 10pt to match aastex7.cls
- **Color cycle**: `Paul Tol's vibrant color scheme <https://personal.sron.nl/~pault/>`_, optimized for color-blind accessibility
- **Ticks**: Inward-facing ticks on all sides with minor ticks enabled
- **Image origin**: Set to ``lower`` (standard in astronomy)
- **Layout**: Uses ``constrained_layout`` for automatic spacing
- **Output**: PDF format at 600 DPI for publication quality

Usage
-----

In Python
~~~~~~~~~

1. Set the style to AAS::

    from brypy import plot

    plot.set_aas_style()

2. When creating a figure, set the figure's width to ``TWO_COLUMN_WIDTH`` or ``ONE_COLUMN_WIDTH``::

    plt.figure(figsize=(plot.TWO_COLUMN_WIDTH, height))

where ``height`` should be tuned manually. Note that the style file includes ``constrained_layout=True``.

In LaTeX
~~~~~~~~

1. Add the following at the top of your document::

    % define consistent figure widths matching actual journal column sizes
    \newlength{\onecol}
    \newlength{\twocol}
    \setlength{\onecol}{3.4in}
    \setlength{\twocol}{7.1in}

2. Then use the following to insert figures::

    \begin{figure*}[htb!]
        \centering
        \includegraphics[width=\twocol]{two.png}
    \end{figure*}

    \begin{figure}[htb!]
        \centering
        \includegraphics[width=\onecol]{single.png}
    \end{figure}

Setting the width this way and using ``\centering`` ensures that it will work for one- and two-column document layouts (e.g., ``preprint`` and ``twocolumn``) where the two-column figures will appear the same and the one-column figures will be the same size but centered in the one-column document layout.
