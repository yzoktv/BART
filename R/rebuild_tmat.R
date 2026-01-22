rebuild_tmat <- function(fit) {
  if (!is.null(fit$tmat_xptr)) return(fit)
  if (is.null(fit$treedraws) || is.null(fit$treedraws$trees)) {
    stop("fit object has no treedraws$trees; cannot rebuild tmat.")
  }

  fit$tmat_xptr <- .Call("C_build_tmat_from_treedraws", fit$treedraws)
  fit
}
