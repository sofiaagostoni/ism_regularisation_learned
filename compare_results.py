"""
Compatibility module: the code now lives in opt_functions/results_lib/ (see its __init__.py),
the interactive cells in explore_sim.py and explore_real.py.

`import compare_results as cr` keeps working: every name of results_lib, public and
private (cr._subset, cr._match_method, cr._REAL_FIELDS, ...), is re-exported here.
Settings: assigning a module constant here (cr.REAL_THEME = "black") does not reach the
code; use opt_functions.results_lib.real.set_real_theme("black").
"""
from opt_functions.results_lib.common import *               # noqa: F401,F403
from opt_functions.results_lib.sim import *                  # noqa: F401,F403
from opt_functions.results_lib.sim_figures import *          # noqa: F401,F403
from opt_functions.results_lib.sim_diagnostics import *      # noqa: F401,F403
from opt_functions.results_lib.real import *                 # noqa: F401,F403
from opt_functions.results_lib.real_figures import *         # noqa: F401,F403
from opt_functions.results_lib.real_diagnostics import *     # noqa: F401,F403

# private names (not exported by *), used by error_vs_int.py, plot_best_methods.py, ...
from opt_functions.results_lib.common import (_as_list, _crop, _fmt, _match_method,  # noqa: F401
                                _method_colors, _plane_label)
from opt_functions.results_lib.sim import _SUM_Y_CACHE, _select, _subset  # noqa: F401
from opt_functions.results_lib.sim_figures import _brightest_window, _fft, _short_labels  # noqa: F401
from opt_functions.results_lib.real import (  # noqa: F401
    _CB_LL, _CB_LR, _CB_UL, _INSET_CORNERS, _PROFILE_COLORS, _PROFILE_LINE, _PROFILE_RAW,
    _REAL_FIELDS, _aspect_window, _diag_dist, _frame_map, _inset_rect, _keep_values,
    _line_profile, _not_converged, _parse_aspect, _pick_real, _real_canvas, _real_cmap,
    _real_colorbar, _real_inset, _real_label, _real_name, _real_norm, _real_pnorm,
    _real_profile_axes, _real_px_nm, _real_scalebar, _real_text, _real_units, _select_real,
    _theme)
from opt_functions.results_lib.real_diagnostics import _ITER_METHODS  # noqa: F401
