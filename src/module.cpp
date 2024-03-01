#include "cyclic_Variational.hpp"

RCPP_MODULE(CYCLICVB){

    class_<CyclicVariationalCausalDiscovery>("CyclicVariationalCausalDiscovery")
    .constructor<mat, string>()
    .method("init_run", &CyclicVariationalCausalDiscovery::init_run)
    .method("run", &CyclicVariationalCausalDiscovery::run)
    .method("get_A", &CyclicVariationalCausalDiscovery::get_A)
    .method("get_pip", &CyclicVariationalCausalDiscovery::get_pip)
    .method("get_mu", &CyclicVariationalCausalDiscovery::get_mu)
    .method("get_noiseParams", &CyclicVariationalCausalDiscovery::get_noiseParams)
    ;
}
