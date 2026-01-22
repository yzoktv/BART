#' Predict BART for a single posterior draw
#'
#' @param object  A pbart/cbart/cgbart model object
#' @param newdata Matrix or data.frame of prediction covariates
#' @param m       Posterior draw index (1 ≤ m ≤ ndpost)
#' @param mc.cores Number of threads for prediction (default 1)
#' @return A list with element `prob.test`, length np vector
#'
#' @export
predict_single_draw <- function(object, newdata, m, mc.cores = 1L) {
  x <- t(as.matrix(newdata))  # cpwbart / cwbart expect p x np
  out <- .Call("C_cpwbart_single", object, x,
               as.integer(m), as.integer(mc.cores))
  # For probit BART, convert yhat to prob using binaryOffset:
  if (!is.null(object$binaryOffset)) {
    prob <- pnorm(as.numeric(out$yhat.test) + object$binaryOffset)
    list(prob.test = prob)
  } else {
    out
  }
}
