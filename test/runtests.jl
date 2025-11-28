using GeneralizedAdditiveModels
using Test
using RDatasets, Plots
using Distributions

#-------------------- Set up data -----------------

df = dataset("datasets", "trees");

#-------------------- Run tests -----------------

@testset "GeneralizedAdditiveModels.jl" begin

    mod = gam("Volume ~ s(Girth, k=10, degree=3) + s(Height, k=10, degree=3)", df)

    p = plotGAM(mod)
    @test p isa Plots.Plot

    # Gamma version

    mod2 = gam("Volume ~ s(Girth, k=10, degree=3) + s(Height, k=10, degree=3)", df; Family = "Gamma", Link = "Log")

    p1 = plotGAM(mod2)
    @test p1 isa Plots.Plot
end

@testset "Bernoulli GAM Tests" begin
    
    # Test 1: Basic Bernoulli GAM with known pattern
    @testset "Basic Bernoulli GAM" begin
        n = 200
        x1 = range(-2, 2, length=n)
        x2 = randn(n)
        
        # Create true nonlinear effect
        f1 = sin.(x1 * π/2)
        f2 = x2.^2 .- 1
        eta = f1 + f2
        p = 1 ./ (1 .+ exp.(-eta))
        y = rand.(Bernoulli.(p))
        
        df = DataFrame(y=y, x1=x1, x2=x2)
        
        # Fit model
        mod = gam("y ~ s(x1, k=8, degree=3) + s(x2, k=8, degree=3)", df; 
                  Family = "Bernoulli", Link = "Logit")
        
        # Basic tests
        @test mod isa GAMData
        @test mod.Family[:Name] == "Bernoulli"
        @test mod.Link[:Name] == "Logit"
        @test all(0 .<= mod.Fitted .<= 1)  # Predictions should be probabilities
        @test length(mod.Fitted) == n
        
        # Test plotting
        p = plotGAM(mod)
        @test p isa Plots.Plot
    end
    
    # Test 2: Edge cases with extreme probabilities
    @testset "Extreme probability cases" begin
        n = 100
        x = range(-5, 5, length=n)
        
        # Create data with extreme probabilities
        eta = 10 * x  # Very strong effect to create extreme probabilities
        p = 1 ./ (1 .+ exp.(-eta))
        y = Float64.(p .> 0.5)  # Deterministic for testing
        
        df = DataFrame(y=y, x=x)
        
        # This should run without numerical errors
        mod = gam("y ~ s(x, k=5, degree=3)", df; 
                  Family = "Bernoulli", Link = "Logit")
        
        @test !any(isnan.(mod.Fitted))
        @test !any(isinf.(mod.Fitted))
        @test all(0 .<= mod.Fitted .<= 1)
    end
    
    # Test 3: Binary validation
    @testset "Binary response validation" begin
        n = 50
        x = randn(n)
        
        # Test with non-binary data (should throw error)
        y_continuous = randn(n)
        df_continuous = DataFrame(y=y_continuous, x=x)
        
        @test_throws AssertionError gam("y ~ s(x, k=5, degree=3)", df_continuous; 
                                        Family = "Bernoulli", Link = "Logit")
        
        # Test with proper binary data (0s and 1s)
        y_binary = rand([0, 1], n)
        df_binary = DataFrame(y=y_binary, x=x)
        
        mod = gam("y ~ s(x, k=5, degree=3)", df_binary; 
                  Family = "Bernoulli", Link = "Logit")
        @test mod isa GAMData
    end
    
    # Test 4: Model diagnostics
    @testset "Model diagnostics" begin
        n = 150
        x = sort(randn(n))
        logit_p = 2*x
        p = 1 ./ (1 .+ exp.(-logit_p))
        y = rand.(Bernoulli.(p))
        
        df = DataFrame(y=y, x=x)
        
        mod = gam("y ~ s(x, k=10, degree=3)", df; 
                  Family = "Bernoulli", Link = "Logit")
        
        # Check diagnostics exist and are reasonable
        @test haskey(mod.Diagnostics, :EDF)
        @test haskey(mod.Diagnostics, :GCV)
        @test mod.Diagnostics[:EDF] > 1  # Should have some smoothing
        @test mod.Diagnostics[:EDF] < 10  # But not too complex
        @test mod.Diagnostics[:GCV] > 0
    end
    
    # Test 5: Convergence behavior
    @testset "PIRLS convergence" begin
        n = 100
        x1 = randn(n)
        x2 = randn(n)
        
        # Simple linear predictor for easier convergence
        eta = 0.5 .+ x1 - 0.5*x2
        p = 1 ./ (1 .+ exp.(-eta))
        y = rand.(Bernoulli.(p))
        
        df = DataFrame(y=y, x1=x1, x2=x2)
        
        # Test with different convergence parameters
        mod1 = gam("y ~ s(x1, k=5, degree=3) + s(x2, k=5, degree=3)", df; 
                   Family = "Bernoulli", Link = "Logit", maxIter = 5)
        
        mod2 = gam("y ~ s(x1, k=5, degree=3) + s(x2, k=5, degree=3)", df; 
                   Family = "Bernoulli", Link = "Logit", maxIter = 50, tol = 1e-8)
        
        @test mod1 isa GAMData
        @test mod2 isa GAMData
        
        # More iterations with tighter tolerance should give similar or better fit
        @test mod2.Diagnostics[:GCV] <= mod1.Diagnostics[:GCV] * 1.1  # Allow 10% tolerance
    end
    
    # Test 6: Comparison with known logistic regression result
    @testset "Linear model special case" begin
        n = 2000
        x = randn(n)
        
        # True linear model
        beta0 = 0.5
        beta1 = 1.5
        eta = beta0 .+ beta1 * x
        p = 1 ./ (1 .+ exp.(-eta))
        y = rand.(Bernoulli.(p))
        
        df = DataFrame(y=y, x=x)
        
        # Fit with many knots to approximate linear function
        mod = gam("y ~ s(x, k=20, degree=3)", df; 
                  Family = "Bernoulli", Link = "Logit")
        
        # Check that predictions are reasonable
        x_test = [-1.0, 0.0, 1.0]
        for xi in x_test
            pred_mat = GeneralizedAdditiveModels.BuildPredictionMatrix([xi], mod.Basis[1], mod.ColMeans[1])
            pred_eta = mod.Coef[1] .+ pred_mat * mod.Coef[mod.CoefIndex[1]]
            pred_p = 1 / (1 + exp(-pred_eta[1]))
            
            true_p = 1 / (1 + exp(-(beta0 + beta1 * xi)))
            
            # Should be reasonably close
            @test abs(pred_p - true_p) < 0.4
        end
    end
end

@testset "Formula Macro Tests" begin
    @testset "SmoothTerm construction" begin
        # Test creating smooth terms with positional arguments
        st1 = s(:x1, 10, 3)
        @test st1 isa SmoothTerm
        @test st1.term.sym == :x1
        @test st1.k == 10
        @test st1.degree == 3

        # Test default values
        st2 = s(:x2)
        @test st2.k == 10  # default
        @test st2.degree == 3  # default
    end

    @testset "Formula parsing from FormulaTerm" begin
        # Test parsing a simple formula with smooth terms
        f = @formula(Volume ~ s(Girth, 10, 3) + s(Height, 5, 2))

        gam_formula = ParseFormula(f)
        @test gam_formula.y == :Volume
        @test nrow(gam_formula.covariates) == 2
        @test gam_formula.covariates.variable[1] == :Girth
        @test gam_formula.covariates.k[1] == 10
        @test gam_formula.covariates.degree[1] == 3
        @test gam_formula.covariates.smooth[1] == true

        @test gam_formula.covariates.variable[2] == :Height
        @test gam_formula.covariates.k[2] == 5
        @test gam_formula.covariates.degree[2] == 2
        @test gam_formula.covariates.smooth[2] == true
    end

    @testset "Formula with mixed smooth and linear terms" begin
        # Test formula with both smooth and linear terms
        f = @formula(Volume ~ s(Girth, 10, 3) + Height)

        gam_formula = ParseFormula(f)
        @test gam_formula.y == :Volume
        @test nrow(gam_formula.covariates) == 2

        # First term is smooth
        @test gam_formula.covariates.variable[1] == :Girth
        @test gam_formula.covariates.smooth[1] == true

        # Second term is linear
        @test gam_formula.covariates.variable[2] == :Height
        @test gam_formula.covariates.smooth[2] == false
        @test gam_formula.covariates.k[2] == 0
        @test gam_formula.covariates.degree[2] == 0
    end

    @testset "GAM fitting with @formula macro" begin
        # Test fitting a GAM using the @formula macro
        f = @formula(Volume ~ s(Girth, 10, 3) + s(Height, 10, 3))

        mod = gam(f, df)
        @test mod isa GAMData
        @test length(mod.Fitted) == nrow(df)

        # Compare with string formula version
        mod_string = gam("Volume ~ s(Girth, k=10, degree=3) + s(Height, k=10, degree=3)", df)

        # Results should be very similar (allowing for numerical precision)
        @test isapprox(mod.Fitted, mod_string.Fitted, rtol=1e-6)
    end

    @testset "GAM with @formula and different families" begin
        # Test with Gamma family
        f = @formula(Volume ~ s(Girth, 10, 3) + s(Height, 10, 3))

        mod_gamma = gam(f, df; Family="Gamma", Link="Log")
        @test mod_gamma isa GAMData
        @test mod_gamma.Family[:Name] == "Gamma"
        @test mod_gamma.Link[:Name] == "Log"

        # Compare with string formula version
        mod_gamma_string = gam("Volume ~ s(Girth, k=10, degree=3) + s(Height, k=10, degree=3)", df;
                               Family="Gamma", Link="Log")
        @test isapprox(mod_gamma.Fitted, mod_gamma_string.Fitted, rtol=1e-6)
    end

    @testset "Bernoulli GAM with @formula" begin
        # Create binary data
        n = 200
        x1 = range(-2, 2, length=n)
        x2 = randn(n)

        # Create true nonlinear effect
        f1 = sin.(x1 * π/2)
        f2 = x2.^2 .- 1
        eta = f1 + f2
        p = 1 ./ (1 .+ exp.(-eta))
        y = rand.(Bernoulli.(p))

        df_bern = DataFrame(y=y, x1=x1, x2=x2)

        # Fit using @formula
        f = @formula(y ~ s(x1, 8, 3) + s(x2, 8, 3))
        mod = gam(f, df_bern; Family="Bernoulli", Link="Logit")

        @test mod isa GAMData
        @test mod.Family[:Name] == "Bernoulli"
        @test all(0 .<= mod.Fitted .<= 1)

        # Compare with string version
        mod_string = gam("y ~ s(x1, k=8, degree=3) + s(x2, k=8, degree=3)", df_bern;
                        Family="Bernoulli", Link="Logit")
        @test isapprox(mod.Fitted, mod_string.Fitted, rtol=1e-6)
    end

    @testset "Plotting GAM fitted with @formula" begin
        f = @formula(Volume ~ s(Girth, 10, 3) + s(Height, 10, 3))
        mod = gam(f, df)

        p = plotGAM(mod)
        @test p isa Plots.Plot
    end
end