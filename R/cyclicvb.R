loadModule("CYCLICVB", TRUE)
#' cyclicvb
#'
#' @param x The observed data following a structural equation model
#' @param noisetype Type of the distribution of the noise in the structural equation model. It needs to be one of the following: 'gaussian','t','laplace','gumbel'. Default :'laplace'.
#' @param seed Set a seed to reproduce results. Default:1
#'
#' @return Inclusion probability of the adjacent matrix
#' @export
#'
#' @examples \dontrun{cyclicvb(matrix(rt(500*10,3),500,10)))
cyclicvb = function(x,noisetype="laplace",seed=1){
  set.seed(seed)
  cycVB = new(CyclicVariationalCausalDiscovery,x,noisetype)
  cycVB$run()
  res = cycVB$get_pip()
  return(res)
}

