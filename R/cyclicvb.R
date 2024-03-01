cyclicvb = function(x,noisetype="laplace"){
  cycVB = new("CYCLICVB",x,noisetype)
  cycVB$run()
  res = cycVB$get_pip()
  return(res)
}

