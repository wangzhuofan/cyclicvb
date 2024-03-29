% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/cyclicvb.R
\name{cyclicvb}
\alias{cyclicvb}
\title{cyclicvb}
\usage{
cyclicvb(x, noisetype = "laplace", seed = 1)
}
\arguments{
\item{x}{The observed data following a structural equation model}

\item{noisetype}{Type of the distribution of the noise in the structural equation model. It needs to be one of the following: 'gaussian','t','laplace','gumbel'. Default :'laplace'.}

\item{seed}{Set a seed to reproduce results. Default:1}
}
\value{
Inclusion probability of the adjacent matrix
}
\description{
cyclicvb
}
