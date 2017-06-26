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
