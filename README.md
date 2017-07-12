# SMC

Unix | CodeCov | License
---- | ------- | -------
[![Travis](https://travis-ci.org/tlienart/SMC.jl.svg?branch=master)](https://travis-ci.org/tlienart/SMC.jl) | [![CodeCov](http://codecov.io/github/tlienart/SMC.jl/coverage.svg?branch=master)](http://codecov.io/github/tlienart/SMC.jl?branch=master) | [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## What's this about

The aim of this package is to offer simple code that illustrate some concepts that are useful in the Sequential Monte Carlo framework.
The primary aim is educational.
If you're looking for something more serious, consider [Lawrence Murray's](http://www.indii.org/about/) excellent package [**Libbi**](http://libbi.org).

## Installation and requirements

Requirements:

* Julia in `[0.6.x]`
* 64-bit architecture

In the Julia REPL:

```julia
Pkg.clone("https://github.com/tlienart/SMC.jl.git")
using SMC
```

## Algorithms implemented

* Kalman Filter and Smoother (2FS) for comparison on Linear Gaussian model
* Particle Filter
* Particle Smoother (FFBS)
* Particle Smoother (Fearnhead's linear complexity method)
* Particle Smoother (BIS(Q), BIS(SQ), aBIS(L))

**Notes**
- FFBS refers to the *Forward Filtering Backward Smoothing*
- 2FS refers to the *Two Filter Formula*
- BIS(Q) refers to the *Backward Information Smoother* algorithm with quadratic complexity
- BIS(SQ) refers to the same algorithm but with a trick reducing the complexity to subquadratic (the estimators are still consistent)
- aBIS(L) refers to the algorithm with independent sampling of mixture labels (see SAPBP) and consequent linear complexity (estimators are *not* consistent)
- SAPBP sequential auxiliary particle belief propagation (contains a method to sample efficiently but approximately from products of mixtures)

## References

(in no particular order)

* Briers, Doucet, Maskell, *Smoothing algorithms for state-space models*, [link](http://www.stats.ox.ac.uk/~doucet/briers_doucet_maskell_smoothingstatespacemodels.pdf), 2009.
* Doucet, Johansen, *A tutorial on particle filtering and smoothing*, [link](http://www.stats.ox.ac.uk/~doucet/doucet_johansen_tutorialPF2011.pdf), 2012.
* Fearnhead, Wyncoll, Tawn, *A sequential smoothing algorithm with linear computational cost*, [link](https://academic.oup.com/biomet/article/97/2/447/219260/A-sequential-smoothing-algorithm-with-linear), 2010.
* Taghavi, *A study of linear complexity particle filter smoothers*, [link](http://publications.lib.chalmers.se/records/fulltext/156741.pdf), 2012.
* Briers, Doucet, Singh, *Sequential auxiliary particle belief propagation*, [link](ieeexplore.ieee.org/document/1591923), 2005.
