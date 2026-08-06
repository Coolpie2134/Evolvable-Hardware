"""
ui - Tk/matplotlib front-ends for the evolvable-hardware backends.

Grouped here so the project root holds only packages and entry points, not a
pile of loose GUI scripts. Launch a front-end as a module from the project
root, e.g.::

    python -m ui.app          # the main single-window GUI
    python -m ui.designer      # standalone manual circuit designer
    python -m ui.concept_gui    # proof-of-concept GA playground

Each front-end imports its siblings relatively (``from .interactive import
...``) and the evolution backends absolutely (``from substrates.nervous import ...``);
running via ``-m`` from the project root puts both on the path.
"""
