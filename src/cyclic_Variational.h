#ifndef CYCLIC_VARIATIONAL_ALGORITHM
#define CYCLIC_VARIATIONAL_ALGORITHM

#include <RcppArmadillo.h>
#include <vector>
#include <map>
#include <string>

using namespace Rcpp;
using namespace arma;
using namespace std;


//' @name CyclicVariationalCausalDiscovery
//' @title CyclicVariationalCausalDiscovery
//' @description This class provides a variational inference algorithm for cyclic structrual equation models
//' @field new Constructor
class CyclicVariationalCausalDiscovery{
public:

  CyclicVariationalCausalDiscovery(mat x_,string noise_type_);

  void init_run();
  void run();

  mat get_A(){
    return A;
  }

  mat get_pip();

  mat get_mu(){
    return mu;
  }

  vec get_noiseParams(){
    return noiseParams;
  }

private:

  // input
  mat x;
  string noiseType;

  // fixed parameters
  int n;
  int p;

  // estimated parameters
  mat A, A_old;
  mat mu, mu_old;
  mat mu_u;
  mat mu_v;
  vec mu_s;
  vec noiseParams;


  // Optimization parameter
  map<string, double> allParams;
  int iterMax;
  int sampleSize;
  double epsilon;
  double stepsize_A,stepsize_mu,stepsize_noise;
  double lambda;
  double c0;
  double c1;
  double pai;
  double mu0;
  double sigma0;
  double sigma;
  double upperbound_A_diag;
  double upperbound_mu_s;
  double lowerbound_mu_s;
  double lowerbound_noise;


  cube u_randomSample;
  cube n_randomSample;
  cube trans_u;


  void update_grad(const mat & A, const mat & mu_u, const vec & mu_s, const mat & mu_v, const vec & noiseParams);
  void update_A();
  void update_mu();
  void update_noise();
  mat stiefelProj(mat G, mat A);
  cube sigmoid_cube(cube T);
  mat sigmoid_mat(mat A);


  mat grad_A, grad_mu_u, grad_mu_v; // internal gradient
  vec grad_mu_s, grad_noiseParams;

};

#endif
