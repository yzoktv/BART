#include <sstream>
#include "tree.h"
#include "treefuns.h"

typedef std::vector<tree> vtree;
typedef std::vector<vtree> vtmat;

extern "C"
SEXP C_build_tmat_from_treedraws(SEXP _itreedraws) {
    Rcpp::List trees(_itreedraws);

    // header string: "nd m p"
    Rcpp::CharacterVector itrees(Rcpp::wrap(trees["trees"]));
    std::string itv(itrees[0]);
    std::stringstream ttss(itv);

    size_t nd, m, p;
    ttss >> nd >> m >> p;

    // allocate vtmat on heap
    vtmat* tmat_all = new vtmat(nd);
    for (size_t i = 0; i < nd; i++) {
        (*tmat_all)[i].resize(m);
        for (size_t j = 0; j < m; j++) {
            ttss >> (*tmat_all)[i][j];
        }
    }

    // return as external pointer; deleted when GC'd
    Rcpp::XPtr<vtmat> tmat_ptr(tmat_all, true);
    return tmat_ptr;
}
