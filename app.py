"""Compatibility launcher for the packaged desktop application.

The implementation lives in :mod:`ui.app`. Keeping this tiny entry point
preserves the established ``python app.py`` and file-association launch paths
without undoing the runtime/substrates/ui package layout.
"""

from ui.app import main


if __name__ == '__main__':
    main()
