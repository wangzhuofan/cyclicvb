#include "cyclic_Variational.h"

// [[Rcpp::depends(RcppArmadillo)]]
//' @name CyclicVariationalCausalDiscovery
//' @title CyclicVariationalCausalDiscovery
//' @description This class provides a variational inference algorithm for cyclic structrual equation models
//' @field new Constructor
CyclicVariationalCausalDiscovery::CyclicVariationalCausalDiscovery(mat x_,string noise_type_){
  allParams.insert ( std::pair<string,double>("iterMax", 500) );
  allParams.insert ( std::pair<string,double>("sampleSize", 1000) );
  allParams.insert ( std::pair<string,double>("epsilon", 0.001) );
  allParams.insert ( std::pair<string,double>("stepsize_A", 0.001) );
  allParams.insert ( std::pair<string,double>("stepsize_mu", 0.001) );
  allParams.insert ( std::pair<string,double>("stepsize_noise", 0.0001) );
  allParams.insert ( std::pair<string,double>("lambda", 2.0/3.0) );
  allParams.insert ( std::pair<string,double>("c0", -0.1) );
  allParams.insert ( std::pair<string,double>("c1", 1.1) );
  allParams.insert ( std::pair<string,double>("pai", 0.1) );
  allParams.insert ( std::pair<string,double>("mu0", 0) );
  allParams.insert ( std::pair<string,double>("sigma0", 1) );
  allParams.insert ( std::pair<string,double>("sigma", 0.1) );
  allParams.insert ( std::pair<string,double>("upperbound_A_diag", -10) );
  allParams.insert ( std::pair<string,double>("upperbound_mu_s", 1) );
  allParams.insert ( std::pair<string,double>("lowerbound_mu_s", -1) );
  allParams.insert ( std::pair<string,double>("lowerbound_noise", 0.0001) );

  x = x_;
  noiseType = noise_type_;

  n = x.n_rows;
  p = x.n_cols;


  // double lambda = 2.0/3.0,c0 = -0.1,c1=1.1,pai=0.1,mu0 = 0,sigma0=1,sigma=0.1;

}

void CyclicVariationalCausalDiscovery::update_grad(const mat & A,
                           const mat & mu_u,
                           const vec & mu_s,
                           const mat & mu_v,
                           const vec & noiseParams){
  mu = mu_u*diagmat(mu_s)*mu_v.t();
  cube binConcrete(p,p,sampleSize),sample_B(p,p,sampleSize);
  binConcrete = trans_u.each_slice()+A;
  sample_B = n_randomSample*sigma;
  sample_B = sample_B.each_slice()+mu;
  binConcrete = sigmoid_cube(binConcrete/lambda);
  cube hardBinConcrete = clamp((c1-c0)*binConcrete+c0,0.0,1.0);
  cube MonteCarloSample = hardBinConcrete%sample_B;
  cube grad_core(p,p,sampleSize);
  cube diff = eye<mat>(p,p)-MonteCarloSample.each_slice();
  mat noiseParamsMat = repmat(noiseParams, 1, n);
  cube Jacobian = diff;
  cube scale_diff = diff;
  Jacobian.each_slice([this](mat& slice) { slice = slice*inv(trans(slice)*slice+0.01 * eye<mat>(p,p)); });
  scale_diff.each_slice([noiseParams](mat& slice) { slice = slice.each_col()/noiseParams; });
  if (noiseType == "gaussian") {
    grad_core = n*Jacobian-scale_diff.each_slice()*(x.t()*x);
  } else if (noiseType == "t") {
    cube temp = diff;
    temp.each_slice([this,noiseParamsMat](mat& slice) { slice = slice * x.t()%(noiseParamsMat + 1)/(pow(slice*x.t(),2)+noiseParamsMat)*x; });
    grad_core = n*Jacobian-temp;
  } else if (noiseType == "gumbel") {
    cube temp = scale_diff;
    temp.each_slice([this,noiseParamsMat](mat& slice) { slice = (1-exp(-slice*x.t()))/noiseParamsMat*x; });
    grad_core = n*Jacobian-temp;
  } else if (noiseType == "laplace") {
    cube temp = diff;
    temp.each_slice([this,noiseParamsMat](mat& slice) { slice = sign(slice*x.t())/noiseParamsMat*x; });
    grad_core = n*Jacobian-temp;
  }

  mat probA = sigmoid_mat(A-lambda*log(-c0/c1));
  // cube gradCore= grad_core(A,mu,noiseType);
  cube AGrad = trans_u;
  AGrad.each_slice([this,A](mat& slice) { slice = exp(-(A+slice)/lambda)%((A>(-lambda*log(-c1/c0)-slice))%(A<(-lambda*log((c1-1)/(1-c0))-slice))); });;
  AGrad = pow(binConcrete,2)/lambda%AGrad;
  mat KL_gradA = (log(probA/pai+1e-10)-log((1-probA)/(1-pai)+1e-10)+(log(sigma0/sigma)+(pow(sigma,2)+pow(mu-mu0,2))/(2*pow(sigma0,2))-0.5))%pow(probA,2)%exp(-A+lambda*log(-c0/c1));
  grad_A = mean(grad_core % sample_B % AGrad, 2);
  grad_A = grad_A+KL_gradA;
  mat KL_gradmu = probA%(mu-mu0)/pow(sigma0,2);
  mat gradmu = mean(grad_core % hardBinConcrete, 2);
  gradmu= gradmu+KL_gradmu;
  grad_mu_u = gradmu*mu_v*diagmat(mu_s);
  grad_mu_v = gradmu.t()*mu_u*diagmat(mu_s);
  // Rcpp::Rcout << "gradient step 1"<<endl;
  grad_mu_s = vec(p, fill::ones);
  for(int i=0;i<p;++i){
    grad_mu_s(i) = accu(gradmu%((mu_u.col(i)*mu_v.col(i).t())));
  }
  // vec gradNoiseParams(p);
  cube y = MonteCarloSample.each_slice()*x.t();
  y = x.t()-y.each_slice();
  cube gradNoiseParamsCube = y;
  mat gradNoiseParamsMat;
  // Rcpp::Rcout<<"this place"<<endl;
  if (noiseType == "gaussian") {
    gradNoiseParamsCube.each_slice([noiseParamsMat](mat& slice) { slice = -0.5*pow(slice,2)/pow(noiseParamsMat,2); });
  } else if (noiseType == "t") {
    gradNoiseParamsCube.each_slice([noiseParamsMat](mat& slice) { slice = 0.5*log(1 + pow(slice,2) / noiseParamsMat) -0.5*
      (noiseParamsMat + 1) / noiseParamsMat %
      pow(slice,2) / (noiseParamsMat + pow(slice,2)); });
  } else if (noiseType == "gumbel") {
    gradNoiseParamsCube.each_slice([noiseParamsMat](mat& slice) { slice = -slice/pow(noiseParamsMat,2)%(1-exp(-(slice/noiseParamsMat))); });
  } else if (noiseType == "laplace") {
    gradNoiseParamsCube.each_slice([noiseParamsMat](mat& slice) { slice = -abs(slice)/pow(noiseParamsMat,2); });
  }
  // Rcpp::Rcout<<gradNoiseParamsCube.n_rows<<" "<<gradNoiseParamsCube.n_cols<<" "<<gradNoiseParamsCube.n_slices<<endl;
  gradNoiseParamsMat = mean(gradNoiseParamsCube,2);
  grad_noiseParams = sum(gradNoiseParamsMat,1);

  if (noiseType == "gaussian") {
    grad_noiseParams += n/2.0/noiseParams;
  } else if (noiseType == "t") {
    vec temp1 = 0.5*(noiseParams+1);
    NumericVector temp_1 = NumericVector(temp1.begin(),temp1.end());
    vec temp2 = 0.5*noiseParams;
    NumericVector temp_2 = NumericVector(temp2.begin(),temp2.end());
    NumericVector temp_3 = digamma(temp_1)-digamma(temp_2);
    vec temp3 = as<vec>(temp_3);
    grad_noiseParams += -0.5*n*(temp3-1.0/noiseParams);
  } else if (noiseType == "gumbel") {
    grad_noiseParams += n/noiseParams;
  } else if (noiseType == "laplace") {
    grad_noiseParams += n/noiseParams;
  }
}


void CyclicVariationalCausalDiscovery::update_A(){
  A = A-stepsize_A*grad_A;
  A.diag() = clamp(diagvec(A),-100,upperbound_A_diag);
}

void CyclicVariationalCausalDiscovery::update_mu(){
  mat uProj = stiefelProj(grad_mu_u,mu_u);
  mat vProj = stiefelProj(grad_mu_v,mu_v);
  mu_u = mu_u -stepsize_mu*uProj;
  mu_s= mu_s-stepsize_mu*grad_mu_s;
  mu_s = clamp(mu_s,lowerbound_mu_s,upperbound_mu_s);
  mu_v = mu_v-stepsize_mu*vProj;
  mat uQ,uR,vQ,vR;
  qr_econ(uQ, uR, mu_u);
  qr_econ(vQ,vR,mu_v);
  mu_u= uQ;
  mu_v= vQ;
  mu = mu_u*diagmat(mu_s)*mu_v.t();
}

void CyclicVariationalCausalDiscovery::update_noise(){
  noiseParams = noiseParams - stepsize_noise*grad_noiseParams;
  noiseParams = clamp(noiseParams,lowerbound_noise,100);
}

void CyclicVariationalCausalDiscovery::run(){
  int iter = 0;
  bool notConverged = true, test1,test2;
  double changePercentA,changePercentMu;
  double normA,normMu;

  init_run();

  do{
    update_grad(A,mu_u,mu_s,mu_v,noiseParams);
    update_A();
    update_mu();
    update_noise();

    changePercentA = norm(A - A_old, "F");
    // Rcpp::Rcout << "changeA"<<changePercentA<<endl;
    changePercentMu = norm(mu - mu_old, "F");
    // Rcpp::Rcout << "changemu"<<changePercentMu<<endl;
    normA = norm(A, "F");
    // Rcpp::Rcout << "normA"<<normA<<endl;
    normMu = norm(mu, "F");
    // Rcpp::Rcout << "normMu"<<normMu<<endl;
    test1 = (changePercentA < epsilon *normA);
    test2 = (changePercentMu < epsilon *normMu);

    // The algorithm converges, either if the relative change is small
    // or if you get zero D matrices.
    if( test1 && test2)  notConverged = false;
    iter++;
    A_old = A;
    mu_old = mu;
  }while(iter < iterMax && notConverged);
}

void CyclicVariationalCausalDiscovery::init_run(){
  iterMax = allParams["iterMax"];
  sampleSize = allParams["sampleSize"];
  epsilon = allParams["epsilon"];
  stepsize_A = allParams["stepsize_A"];
  stepsize_mu = allParams["stepsize_mu"];
  stepsize_noise = allParams["stepsize_noise"];
  lambda = allParams["lambda"];
  c0 = allParams["c0"];
  c1 = allParams["c1"];
  pai = allParams["pai"];
  mu0 = allParams["mu0"];
  sigma0 = allParams["sigma0"];
  sigma = allParams["sigma"];
  upperbound_A_diag = allParams["upperbound_A_diag"];
  upperbound_mu_s = allParams["upperbound_mu_s"];
  lowerbound_mu_s = allParams["lowerbound_mu_s"];
  lowerbound_noise = allParams["lowerbound_noise"];

  
  cube u_randomSample = randu<cube>(p,p,sampleSize);
  cube n_randomSample = randn<cube>(p,p,sampleSize);
  trans_u = u_randomSample/(1-u_randomSample);

  mat tmp1 = mat(p, p, fill::ones);
  A = -2 * tmp1;
  mat tmp0 = mat(p, p, fill::zeros);
  mu = tmp0;
  svd(mu_u, mu_s, mu_v, tmp0);
  noiseParams = vec(p, fill::ones);
  A_old = tmp1;
  mu_old = tmp0;
}

cube CyclicVariationalCausalDiscovery::sigmoid_cube(cube x){
  cube res = 1/(1+exp(-x));
  return res;
}
mat CyclicVariationalCausalDiscovery::sigmoid_mat(mat x){
  mat res = 1/(1+exp(-x));
  return res;
}

mat CyclicVariationalCausalDiscovery::stiefelProj(mat G, mat A){
  mat tmp;
  tmp = A.t() * G;
  tmp = A * (tmp  + tmp.t());
  tmp = G - 0.5 * tmp;
  return tmp;
}

mat CyclicVariationalCausalDiscovery::get_pip(){
  mat pip = sigmoid_mat(A-lambda*log(-c0/c1));
  return pip;
}
