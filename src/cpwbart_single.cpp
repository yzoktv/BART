// src/cpwbart_single.cpp
//
// Single-draw probit BART prediction using pre-parsed trees.
//
//  - Expects `fit` to be a pbart object with:
//      * fit$treedraws$trees     : header string with nd, m, p
//      * fit$treedraws$cutpoints : list of cutpoints
//      * fit$tmat_xptr           : XPtr< std::vector< std::vector<tree> > >
//  - Evaluates only ONE posterior draw (index `draw`)
//  - Returns yhat.test as a 1 x np numeric matrix
//

#include <sstream>
#include "tree.h"
#include "treefuns.h"

typedef std::vector<tree> vtree;
typedef std::vector<vtree> vtmat;

// helper: compute predictions for a single draw index
static void getpred_single_from_tmat(
    int draw_index,          // 0-based index into tmat (draw)
    size_t p,                // number of predictors
    size_t m,                // number of trees in sum
    size_t np,               // number of prediction points
    xinfo &xi,
    vtmat &tmat,
    double *px,
    Rcpp::NumericMatrix &yhat  // 1 x np, fill row 0
) {
    double *fptemp = new double[np];

    vtree &trees_this_draw = tmat[draw_index];

    for (size_t j = 0; j < m; j++) {
        fit(trees_this_draw[j], xi, p, np, px, fptemp);
        for (size_t k = 0; k < np; k++) {
            yhat(0, k) += fptemp[k];
        }
    }

    delete [] fptemp;
}

// ---------------------------------------------------------------------
//  C_cpwbart_single
//
//  Args:
//    _ifit  : full pbart fit object (R list)
//    _ix    : prediction matrix (same orientation as cpwbart: p x np)
//    _idraw : which posterior draw (1..nd)
//    _itc   : thread count (currently ignored; placeholder)
// ---------------------------------------------------------------------

extern "C"
SEXP C_cpwbart_single(
    SEXP _ifit,
    SEXP _ix,
    SEXP _idraw,
    SEXP _itc
) {
    Rcpp::List fit(_ifit);

    // 1) Extract binary forest via external pointer -------------------
    if (fit.containsElementNamed("tmat_xptr") == FALSE) {
        Rcpp::stop("fit object has no 'tmat_xptr'; "
                   "did you modify pbart to store the parsed trees?");
    }

    Rcpp::XPtr<vtmat> tmat_ptr(fit["tmat_xptr"]);
    if (tmat_ptr.get() == NULL) {
        Rcpp::stop("tmat_xptr is NULL");
    }
    vtmat &tmat = *tmat_ptr;      // vector< vector<tree> >
    size_t nd_parsed = tmat.size();

    // 2) Extract treedraws metadata (nd, m, p) & cutpoints ------------
    Rcpp::List trees(fit["treedraws"]);

    // thread count (currently unused, but keep for API symmetry)
    int tc = Rcpp::as<int>(_itc);
    (void)tc;

    // header string: "nd m p"
    Rcpp::CharacterVector itrees(Rcpp::wrap(trees["trees"]));
    std::string itv(itrees[0]);
    std::stringstream ttss(itv);

    size_t nd_hdr, m_hdr, p_hdr;
    ttss >> nd_hdr >> m_hdr >> p_hdr;

    // sanity check: number of draws from header vs parsed
    if (nd_hdr != nd_parsed) {
        Rcpp::Rcout << "WARNING: nd from header (" << nd_hdr
                    << ") != nd from tmat_xptr (" << nd_parsed << ")\n";
    }
    size_t nd = nd_parsed;

    // read requested draw index (1-based from R)
    int draw = Rcpp::as<int>(_idraw);
    if (draw < 1 || static_cast<size_t>(draw) > nd) {
        Rcpp::stop("draw index out of range: got %d, but nd = %d",
                   draw, (int)nd);
    }
    int draw_index = draw - 1; // 0-based for C++

    // number of trees per draw: from tmat
    if (tmat[draw_index].empty()) {
        Rcpp::stop("tmat[draw_index] is empty (no trees stored?)");
    }
    size_t m = tmat[draw_index].size();

    // process cutpoints
    Rcpp::List ixi(Rcpp::wrap(trees["cutpoints"]));
    size_t p = ixi.size();
    if (p_hdr != p) {
        Rcpp::Rcout << "WARNING: p from header (" << p_hdr
                    << ") != length(cutpoints) (" << p << ")\n";
    }

    xinfo xi;
    xi.resize(p);
    for (size_t i = 0; i < p; i++) {
        Rcpp::NumericVector cutv(ixi[i]);
        xi[i].resize(cutv.size());
        std::copy(cutv.begin(), cutv.end(), xi[i].begin());
    }

    // 3) Process x (same orientation as cpwbart: p x np) --------------
    Rcpp::NumericMatrix xpred(_ix);
    size_t np = xpred.ncol();  // number of prediction points

    double *px = &xpred(0, 0);

    // 4) Get predictions for this single draw -------------------------
    Rcpp::NumericMatrix yhat(1, np);
    std::fill(yhat.begin(), yhat.end(), 0.0);

    getpred_single_from_tmat(draw_index, p, m, np, xi, tmat, px, yhat);

    // 5) Return list --------------------------------------------------
    Rcpp::List ret;
    ret["yhat.test"] = yhat;
    return ret;
}
