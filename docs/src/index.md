# GeneralizedAdditiveModels.jl

**GeneralizedAdditiveModels.jl** provides a flexible and efficient framework for fitting, interpreting, and visualizing *generalized additive models* (GAMs) in Julia.

---

## What Is a GAM?

A *generalized additive model* (GAM) is a semi-parametric regression model that captures smooth and nonlinear relationships between predictors and a response variable. Formally, it can be written as:

```math
g(\mathbb{E}[Y \mid X, \theta]) = \beta_0 + f_1(x_1) + f_2(x_2) + \dots + f_n(x_n)
```

where:

- \( Y \) is the response (dependent variable)
- \( X \) is the design matrix containing the predictor (independent) variables
- \( g(\cdot) \) is the link function
- each \( f_j(\cdot) \) is a *smooth* function of one of the predictors \( x_j \), typically represented using splines or other basis functions
- \( \beta_0 \) is the intercept term

A GAM is thus an extension of the generalized linear model (GLM).  
The key difference is that a GAM uses *smooth*, flexible functions of the predictors, while maintaining an *additive* structure.  
This combination allows GAMs to model complex, nonlinear effects while remaining interpretable — a valuable property in modern statistical modeling.

---

## Installation

GeneralizedAdditiveModels.jl is not yet registered in the Julia General registry.  
To install the package, clone it directly from the repository:

```julia
using Pkg
Pkg.add(url="https://github.com/hendersontrent/GAM.jl")
```

---

## Contributing

**GeneralizedAdditiveModels.jl** is under active development, and contributions are very welcome!

- If you’ve found a bug or want to propose a feature, please [open an issue](https://github.com/hendersontrent/GAM.jl/issues).
- If your idea gets positive feedback, feel free to submit a pull request.
- If you’re unsure where to start, you can also browse the open issues and pick one that interests you.

---
