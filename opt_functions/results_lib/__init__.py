"""
Analysis of the results (formerly all in compare_results.py, which still re-exports
everything for the scripts that import it):

    common            display names, metric labels, to_img / _crop, method matching
    sim               simulated results: loading, filters, text and colour-coded tables,
                      result files of one row (load_rec, sum_measurements, get_image)
    sim_figures       reconstructions, profiles, paper figure, all images, single images
    sim_diagnostics   PSNR per iteration, functional, convergence check, parameter search
    real              real data: loading, selection, labels, publicity drawing helpers
    real_figures      plot_real, plot_real_single, raw / detectors, split / bands figures
    real_diagnostics  convergence of the real runs, residual whiteness

Interactive cells: explore_sim.py and explore_real.py (repository root).
"""
