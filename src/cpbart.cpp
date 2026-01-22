// cpbart.cpp  -- probit BART with tree storage + tmat_xptr

/*
 *  BART: Bayesian Additive Regression Trees
 *  Copyright (C) 2017 Robert McCulloch and Rodney Sparapani
 *
 *  GPL-2 or later
 */

#include <ctime>
#include <Rcpp.h>
using namespace Rcpp;

#include "tree.h"
#include "treefuns.h"
#include "info.h"
#include "bartfuns.h"
#include "bd.h"
#include "bart.h"
#include "rtnorm.h"

typedef std::vector<tree> vtree;
typedef std::vector<vtree> vtmat;

#define TRDRAW(a, b) trdraw(a, b)
#define TEDRAW(a, b) tedraw(a, b)

RcppExport SEXP cpbart(
   SEXP _in,            //number of observations in training data
   SEXP _ip,            //dimension of x
   SEXP _inp,           //number of observations in test data
   SEXP _ix,            //x, train,  pxn (transposed so rows are contiguous in memory)
   SEXP _iy,            //y, train,  nx1
   SEXP _ixp,           //x, test, pxnp (transposed so rows are contiguous in memory)
   SEXP _im,            //number of trees
   SEXP _inc,           //number of cut points
   SEXP _ind,           //number of kept draws (except for thinnning ..)
   SEXP _iburn,         //number of burn-in draws skipped
   SEXP _ipower,
   SEXP _ibase,
   SEXP _binaryOffset,
   SEXP _itau,
   SEXP _idart,
   SEXP _itheta,
   SEXP _iomega,
   SEXP _igrp,
   SEXP _ia,
   SEXP _ib,
   SEXP _irho,
   SEXP _iaug,
   SEXP _inkeeptrain,
   SEXP _inkeeptest,
   SEXP _inkeeptreedraws,
   SEXP _inprintevery,
   SEXP _Xinfo
) {
   //--------------------------------------------------
   // process args
   size_t n  = Rcpp::as<int>(_in);
   size_t p  = Rcpp::as<int>(_ip);
   size_t np = Rcpp::as<int>(_inp);

   Rcpp::NumericVector  xv(_ix);
   double *ix = &xv[0];

   Rcpp::IntegerVector  yv(_iy); // binary
   int *iy = &yv[0];

   Rcpp::NumericMatrix xpv(_ixp);
   double *ixp = nullptr;
   if (np > 0) ixp = &xpv[0];

   size_t m = Rcpp::as<int>(_im);

   Rcpp::IntegerVector _nc(_inc);
   int *numcut = &_nc[0];

   size_t nd   = Rcpp::as<int>(_ind);
   size_t burn = Rcpp::as<int>(_iburn);

   double mybeta       = Rcpp::as<double>(_ipower);
   double alpha        = Rcpp::as<double>(_ibase);
   double binaryOffset = Rcpp::as<double>(_binaryOffset);
   double tau          = Rcpp::as<double>(_itau);

   bool dart = (Rcpp::as<int>(_idart) == 1);

   double a    = Rcpp::as<double>(_ia);
   double b    = Rcpp::as<double>(_ib);
   double rho  = Rcpp::as<double>(_irho);
   bool aug    = (Rcpp::as<int>(_iaug) == 1);
   double theta = Rcpp::as<double>(_itheta);
   double omega = Rcpp::as<double>(_iomega);

   Rcpp::IntegerVector _grp(_igrp);
   int *grp = &_grp[0];
   (void)grp; // currently unused; avoid warning

   size_t nkeeptrain    = Rcpp::as<int>(_inkeeptrain);
   size_t nkeeptest     = Rcpp::as<int>(_inkeeptest);
   size_t nkeeptreedraws= Rcpp::as<int>(_inkeeptreedraws);
   size_t printevery    = Rcpp::as<int>(_inprintevery);

   Rcpp::NumericMatrix Xinfo(_Xinfo);

   // return structures
   Rcpp::NumericMatrix trdraw(nkeeptrain, n);
   Rcpp::NumericMatrix tedraw(nkeeptest,  np);
   Rcpp::NumericMatrix varprb(nkeeptreedraws, p);
   Rcpp::IntegerMatrix varcnt(nkeeptreedraws, p);

   // RNG + model
   arn  gen;
   bart bm(m);

   if (Xinfo.size() > 0) {
      xinfo _xi;
      _xi.resize(p);
      for (size_t i = 0; i < p; i++) {
         _xi[i].resize(numcut[i]);
         for (size_t j = 0; j < (size_t)numcut[i]; j++)
            _xi[i][j] = Xinfo(i, j);
      }
      bm.setxinfo(_xi);
   }

   //--------------------------------------------------
   // latent z
   double* iz = new double[n];

   //--------------------------------------------------
   // tree text stream
   std::stringstream treess;
   treess.precision(10);
   treess << nkeeptreedraws << " " << m << " " << p << std::endl;

   // dart iter stats
   std::vector<double> ivarprb(p, 0.);
   std::vector<size_t> ivarcnt(p, 0);

   printf("*****Into main of pbart\n");

   size_t skiptr, skipte, skiptreedraws;
   if (nkeeptrain) skiptr = nd / nkeeptrain;
   else            skiptr = nd + 1;

   if (nkeeptest) skipte = nd / nkeeptest;
   else           skipte = nd + 1;

   if (nkeeptreedraws) skiptreedraws = nd / nkeeptreedraws;
   else                skiptreedraws = nd + 1;

   // how many draws are we storing as trees
   size_t nkeeptreedraws_eff = nkeeptreedraws;

   vtmat *tmat_all = nullptr;
   if (nkeeptreedraws_eff > 0) {
      tmat_all = new vtmat(nkeeptreedraws_eff);  // vector< vector<tree> >
   }
   size_t itree_draw = 0;

   //--------------------------------------------------
   // print args
   printf("*****Data:\n");
   printf("data:n,p,np: %zu, %zu, %zu\n", n, p, np);
   printf("y1,yn: %d, %d\n", iy[0], iy[n-1]);
   printf("x1,x[n*p]: %lf, %lf\n", ix[0], ix[n*p-1]);
   if (np) printf("xp1,xp[np*p]: %lf, %lf\n", ixp[0], ixp[np*p-1]);
   printf("*****Number of Trees: %zu\n", m);
   printf("*****Number of Cut Points: %d ... %d\n", numcut[0], numcut[p-1]);
   printf("*****burn and ndpost: %zu, %zu\n", burn, nd);
   printf("*****Prior:mybeta,alpha,tau: %lf,%lf,%lf\n",
          mybeta, alpha, tau);
   printf("*****binaryOffset: %lf\n", binaryOffset);
   cout << "*****Dirichlet:sparse,theta,omega,a,b,rho,augment: "
             << dart << ',' << theta << ',' << omega << ',' << a << ','
             << b << ',' << rho << ',' << aug << std::endl;
   printf("*****nkeeptrain,nkeeptest,nkeeptreedraws: %zu,%zu,%zu\n",
          nkeeptrain, nkeeptest, nkeeptreedraws);
   printf("*****printevery: %zu\n", printevery);
   printf("*****skiptr,skipte,skiptreedraws: %zu,%zu,%zu\n",
          skiptr, skipte, skiptreedraws);

   //--------------------------------------------------
   bm.setprior(alpha, mybeta, tau);
   bm.setdata(p, n, ix, iz, numcut);
   bm.setdart(a, b, rho, aug, dart, theta, omega);

   // init latent z
   for (size_t k = 0; k < n; k++) {
      if (iy[k] == 0) iz[k] = -rtnorm(0.,  binaryOffset, 1., gen);
      else            iz[k] =  rtnorm(0., -binaryOffset, 1., gen);
   }

   //--------------------------------------------------
   // temporary storage for test fits
   double* fhattest = 0;
   if (np) fhattest = new double[np];

   //--------------------------------------------------
   // MCMC
   printf("\nMCMC\n");
   size_t trcnt = 0;
   size_t tecnt = 0;
   bool keeptest, keeptreedraw;

   time_t tp;
   int time1 = time(&tp);
   xinfo& xi = bm.getxinfo();

   size_t total = nd + burn;
   for (size_t i = 0; i < total; i++) {
      if (i % printevery == 0)
         printf("done %zu (out of %zu)\n", i, total);

      if (i == (burn / 2) && dart) bm.startdart();

      // draw trees
      bm.draw(1., gen);

      // update z
      for (size_t k = 0; k < n; k++) {
         if (iy[k] == 0)
            iz[k] = -rtnorm(-bm.f(k),  binaryOffset, 1., gen);
         else
            iz[k] =  rtnorm( bm.f(k), -binaryOffset, 1., gen);
      }

      if (i >= burn) {
         // keep train fits
         if (nkeeptrain && (((i - burn + 1) % skiptr) == 0)) {
            for (size_t k = 0; k < n; k++)
               TRDRAW(trcnt, k) = bm.f(k);
            trcnt += 1;
         }

         // keep test fits
         keeptest = nkeeptest && (((i - burn + 1) % skipte) == 0) && np;
         if (keeptest) {
            bm.predict(p, np, ixp, fhattest);
            for (size_t k = 0; k < np; k++)
               TEDRAW(tecnt, k) = fhattest[k];
            tecnt += 1;
         }

         // keep tree draws
         keeptreedraw = nkeeptreedraws &&
                        (((i - burn + 1) % skiptreedraws) == 0);
         if (keeptreedraw) {
            // write trees to text
            for (size_t j = 0; j < m; j++) {
               treess << bm.gettree(j);
            }

            // also copy into binary forest
            if (tmat_all && itree_draw < nkeeptreedraws_eff) {
               (*tmat_all)[itree_draw].resize(m);
               for (size_t j = 0; j < m; j++) {
                  (*tmat_all)[itree_draw][j] = bm.gettree(j);
               }
               itree_draw++;
            }

            // variable usage stats
            ivarcnt = bm.getnv();
            ivarprb = bm.getpv();
            size_t k = (i - burn) / skiptreedraws;
            for (size_t j = 0; j < p; j++) {
               varcnt(k, j) = ivarcnt[j];
               varprb(k, j) = ivarprb[j];
            }
         }
      }
   }

   int time2 = time(&tp);
   printf("time: %ds\n", time2 - time1);
   printf("check counts\n");
   printf("trcnt,tecnt: %zu,%zu\n", trcnt, tecnt);

   if (fhattest) delete[] fhattest;
   delete[] iz;

   //--------------------------------------------------
   // return to R
   Rcpp::List ret;
   ret["yhat.train"] = trdraw;
   ret["yhat.test"]  = tedraw;
   ret["varcount"]   = varcnt;
   ret["varprob"]    = varprb;

   Rcpp::List xiret(xi.size());
   for (size_t i = 0; i < xi.size(); i++) {
      Rcpp::NumericVector vtemp(xi[i].size());
      std::copy(xi[i].begin(), xi[i].end(), vtemp.begin());
      xiret[i] = Rcpp::NumericVector(vtemp);
   }

   Rcpp::List treesL;
   treesL["cutpoints"] = xiret;
   treesL["trees"]     = Rcpp::CharacterVector(treess.str());
   ret["treedraws"]    = treesL;

   // attach binary forest as external pointer
   if (tmat_all != nullptr) {
      Rcpp::XPtr<vtmat> tmat_ptr(tmat_all, true); // delete when GC'd
      ret["tmat_xptr"] = tmat_ptr;
   } else {
      ret["tmat_xptr"] = R_NilValue;
   }

   return ret;
}
