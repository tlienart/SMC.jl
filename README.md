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

## Things implemented

* Kalman Filter and Smoother (for comparison on Linear Gaussian model)
* Particle Filter for HMM
* Particle Smoother (FFBS)

**Notes**
- FFBS refers to the *Forward Filtering Backward Smoothing*
- 2FS refers to the *Two Filter Formula*

## References

(in no particular order)

* Briers, Doucet, Maskell, *Smoothing algorithms for state-space models*, [link](http://www.stats.ox.ac.uk/~doucet/briers_doucet_maskell_smoothingstatespacemodels.pdf), 2009.
* Doucet, Johansen, *A tutorial on particle filtering and smoothing*, [link](http://www.stats.ox.ac.uk/~doucet/doucet_johansen_tutorialPF2011.pdf), 2012.
