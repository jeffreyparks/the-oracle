# Bayesian Modeling and Forecasting — Syllabus

*Seed skill graph for The Oracle. Source of truth: `data/packs/objectives/*.yaml` plus
`data/packs/domains/bayesian_forecasting.yaml`. (The original monolithic file was
split in Phase 0 and removed once superseded; it remains in git history.)*

**70 objectives · 10 modules · 15 tracked misconceptions · ~65 hours of study · max prerequisite depth 22**

## What this course is for

You finish able to build a probabilistic forecast that a business can act on, and able to
prove it is good. "Prove" means backtested on a rolling origin, scored with proper rules,
calibration checked, and connected to a loss function.

The graph is deliberately forecast-heavy. General Bayesian modeling is the first two thirds;
time series, evaluation, and decisions are the last third and the capstone.

### Design rules

- **Every objective is a DAG node.** Prerequisites are hard edges. The Architect agent may
  reorder within a module, never across an edge.
- **Bloom level drives item type.** `remember`/`understand` objectives get recall and
  explanation items. `apply` gets code and computation. `analyze`/`evaluate` gets diagnosis
  and judgement. `create` gets build-and-defend tasks.
- **Difficulty 1-5** is cognitive load, not length. `est_minutes` is length.
- **Assessment stems reveal mastery, not recall.** Most stems are diagnose-or-defend, because
  a student who has only memorized can restate a definition but cannot say what to do when
  R-hat is 1.08.
- **Misconceptions are first-class.** Each names a wrong mental model, the objectives it
  attaches to, and one question that forces it into the open.

## Module map

1. **Probability Foundations and the Simulation Mindset** — 8 objectives, ~6.7 h. Build fluent probability language and the habit of answering questions by simulation, not formulas alone.
2. **Bayes' Theorem, Conjugacy, and Reading a Posterior** — 6 objectives, ~4.9 h. Update beliefs correctly by hand, and report posteriors in language that answers the real question.
3. **Priors, Generative Models, and Model Validation** — 6 objectives, ~5.5 h. Choose defensible priors, write full generative models, and prove the implementation is correct before trusting it.
4. **Computation: MCMC and Variational Inference** — 8 objectives, ~7.7 h. Understand and run the algorithms that make non-conjugate Bayesian inference possible.
5. **Diagnostics and Posterior Predictive Checking** — 7 objectives, ~5.8 h. Decide whether a fit is trustworthy, and whether the model can reproduce the features of the data you care about.
6. **Bayesian Regression: GLMs, Robustness, and Smooths** — 6 objectives, ~5.5 h. Model a conditional mean honestly across outcome types, nonlinearity, outliers, and imperfect data.
7. **Hierarchical and Multilevel Models** — 6 objectives, ~5.7 h. Share strength across groups, and fix the sampling geometry that hierarchical models create.
8. **Model Comparison and Averaging** — 4 objectives, ~3.5 h. Compare models on expected out-of-sample predictive accuracy, and combine them rather than over-trusting one.
9. **Bayesian Time Series and Structural Forecasting** — 10 objectives, ~9.6 h. Model time-dependent data with state space, seasonal, GP, and hierarchical structure, and produce honest predictive distributions.
10. **Forecast Evaluation, Decisions, and the Full Workflow** — 9 objectives, ~10.6 h. Prove a forecast is good with proper scores and calibration, turn it into decisions, and run the whole workflow end to end.


## Module 1: Probability Foundations and the Simulation Mindset

**Goal.** Build fluent probability language and the habit of answering questions by simulation, not formulas alone.

**Load.** 8 objectives, ~6.7 hours.

### `prob_sample_space_events` — Sample Spaces, Events, and Probability Axioms

State the Kolmogorov axioms and use them to compute probabilities of compound events.

- Bloom: **understand** · Difficulty: **1/5** · Est: **40 min**
- Prerequisites: *none (entry point)*
- Assessment stems:
  1. Given a two-dice experiment, write the sample space and compute P(sum is prime).
  1. A student claims P(A or B) = P(A) + P(B) for any A, B. Give a counterexample and the correct rule.
  1. Show that P(A^c) = 1 - P(A) follows from the axioms.

### `prob_conditional_independence` — Conditional Probability and Independence

Compute conditional probabilities and distinguish independence from conditional independence.

- Bloom: **apply** · Difficulty: **2/5** · Est: **50 min**
- Prerequisites: `prob_sample_space_events`
- Assessment stems:
  1. Two events are independent but not conditionally independent given C. Construct such a case.
  1. From a 2x2 contingency table, compute P(disease | positive test) and interpret it.
  1. Explain why 'A and B are independent' does not imply 'A and B are independent given C'.

### `prob_random_variables` — Random Variables, PMFs, PDFs, and CDFs

Distinguish discrete and continuous random variables and move between pmf/pdf, CDF, and quantiles.

- Bloom: **understand** · Difficulty: **2/5** · Est: **50 min**
- Prerequisites: `prob_sample_space_events`
- Assessment stems:
  1. Given a pdf f(x) = cx on [0,2], find c, the CDF, and the median.
  1. Explain why a pdf value may exceed 1 but a pmf value may not.
  1. Sketch the CDF of a mixture of a point mass at 0 and a Normal, and name where it is discontinuous.

### `prob_expectation_variance` — Expectation, Variance, and Moments

Compute and interpret expectations, variances, covariances, and the law of total expectation/variance.

- Bloom: **apply** · Difficulty: **2/5** · Est: **50 min**
- Prerequisites: `prob_random_variables`
- Assessment stems:
  1. Use the law of total variance to decompose the variance of a hierarchical two-stage draw.
  1. Show E[XY] != E[X]E[Y] when X and Y are dependent, with a concrete example.
  1. A forecaster reports a mean of 100 and sd of 40 for a strictly positive quantity. What does this imply about skew?

### `prob_common_distributions` — The Common Distribution Families and Their Stories

Match Bernoulli, Binomial, Poisson, Negative Binomial, Normal, Student-t, Beta, Gamma, and Dirichlet to the data-generating stories they encode.

- Bloom: **analyze** · Difficulty: **2/5** · Est: **60 min**
- Prerequisites: `prob_random_variables`, `prob_expectation_variance`
- Assessment stems:
  1. Counts of website errors per hour are overdispersed relative to Poisson. Name a better family and justify it from its story.
  1. Why is the Beta the natural family for an unknown proportion, and the Gamma for an unknown rate?
  1. Given four datasets, assign a likelihood family to each and defend the choice in one sentence.

### `prob_joint_marginal_transform` — Joint Distributions, Marginalization, and Change of Variables

Marginalize and condition in joint distributions and apply the Jacobian change-of-variables rule.

- Bloom: **apply** · Difficulty: **3/5** · Est: **60 min**
- Prerequisites: `prob_expectation_variance`, `prob_common_distributions`
- Assessment stems:
  1. Derive the density of Y = log X when X is LogNormal, showing the Jacobian term.
  1. Given a bivariate Normal, derive the conditional mean and variance of X1 given X2.
  1. Explain why marginalizing a nuisance parameter is not the same as fixing it at its mean.

### `prob_mc_integration` — Monte Carlo Integration and the Law of Large Numbers

Approximate expectations and tail probabilities with simulation and quantify Monte Carlo error.

- Bloom: **apply** · Difficulty: **2/5** · Est: **50 min**
- Prerequisites: `prob_expectation_variance`
- Assessment stems:
  1. Estimate P(X > 3) for a Student-t(4) by simulation and report the Monte Carlo standard error.
  1. How many draws are needed so the MC standard error of a mean is below 1% of the posterior sd?
  1. Why can any posterior summary be computed from draws without ever writing down the posterior density?

### `tool_python_prob_stack` — Tooling: Python Probability Stack Setup

Set up and run a reproducible Bayesian workflow environment with NumPy, SciPy, a PPL, and ArviZ.

- Bloom: **apply** · Difficulty: **1/5** · Est: **40 min**
- Prerequisites: `prob_mc_integration`
- Assessment stems:
  1. Install a PPL and ArviZ in a fresh environment, then draw and plot 10,000 Beta(2,5) samples.
  1. Set a seed so a sampling run is byte-reproducible, and show the evidence.
  1. Convert raw sampler output into an ArviZ InferenceData object and inspect its groups.


## Module 2: Bayes' Theorem, Conjugacy, and Reading a Posterior

**Goal.** Update beliefs correctly by hand, and report posteriors in language that answers the real question.

**Load.** 6 objectives, ~4.9 hours.

### `bayes_theorem` — Bayes' Theorem and the Anatomy of the Posterior

Derive Bayes' theorem and identify prior, likelihood, evidence, and posterior in a worked inference.

- Bloom: **understand** · Difficulty: **2/5** · Est: **50 min**
- Prerequisites: `prob_conditional_independence`, `prob_common_distributions`
- Assessment stems:
  1. A test has 99% sensitivity and 95% specificity; prevalence is 0.1%. Compute P(disease | positive) and explain the surprise.
  1. Label prior, likelihood, and normalizing constant in a written-out posterior, and say which one does not affect the shape.
  1. Show algebraically why the posterior is unchanged if the likelihood is multiplied by a constant.

### `bayes_likelihood_function` — The Likelihood Function and Its Interpretation

Write a likelihood for an i.i.d. sample and explain why it is a function of parameters, not a probability over them.

- Bloom: **understand** · Difficulty: **2/5** · Est: **45 min**
- Prerequisites: `bayes_theorem`
- Assessment stems:
  1. Write the log-likelihood for n Bernoulli trials and find its maximizer.
  1. A student says 'the likelihood of theta = 0.5 is 0.03, so there is a 3% chance theta is 0.5.' Correct them.
  1. Explain why the likelihood need only be specified up to a constant in theta.

### `bayes_conjugacy` — Conjugate Priors and Closed-Form Updating

Derive and apply Beta-Binomial, Gamma-Poisson, and Normal-Normal conjugate updates.

- Bloom: **apply** · Difficulty: **3/5** · Est: **60 min**
- Prerequisites: `bayes_likelihood_function`, `prob_joint_marginal_transform`
- Assessment stems:
  1. Starting from Beta(2,2) and observing 7 successes in 10 trials, give the posterior and its 90% interval.
  1. Derive the Normal-Normal posterior mean as a precision-weighted average of prior mean and data mean.
  1. Show how a Gamma-Poisson posterior behaves as exposure time grows without bound.

### `bayes_sequential_updating` — Sequential Updating and Exchangeability

Show that batch and sequential Bayesian updates agree, and explain exchangeability as the modeling assumption behind i.i.d. likelihoods.

- Bloom: **analyze** · Difficulty: **3/5** · Est: **50 min**
- Prerequisites: `bayes_conjugacy`
- Assessment stems:
  1. Update a Beta prior one observation at a time and in one batch; prove numerically the posteriors match.
  1. Give a dataset where exchangeability plainly fails and say what structure must be added.
  1. State de Finetti's theorem informally and connect it to hierarchical models.

### `bayes_credible_vs_confidence` — Credible Intervals vs Confidence Intervals

Contrast Bayesian credible intervals (HDI, equal-tailed) with frequentist confidence intervals in coverage meaning.

- Bloom: **analyze** · Difficulty: **3/5** · Est: **45 min**
- Prerequisites: `bayes_conjugacy`, `prob_mc_integration`
- Assessment stems:
  1. Compute both an HDI and an equal-tailed interval for a skewed posterior and explain when they differ.
  1. A colleague says 'the 95% CI means there is 95% probability the parameter is inside.' When is that true and when not?
  1. Design a simulation showing a credible interval's frequentist coverage under a misspecified prior.

### `bayes_posterior_summaries` — Posterior Summaries and Reporting

Choose and report point estimates, intervals, and probability statements that answer the actual question asked.

- Bloom: **evaluate** · Difficulty: **2/5** · Est: **45 min**
- Prerequisites: `bayes_credible_vs_confidence`
- Assessment stems:
  1. Report P(effect > 0), P(effect > a practical threshold), and the HDI for one posterior, and say which the stakeholder needs.
  1. When is the posterior mode a poor summary? Give a distribution where it misleads.
  1. Rewrite a p-value-style conclusion as an honest posterior statement.


## Module 3: Priors, Generative Models, and Model Validation

**Goal.** Choose defensible priors, write full generative models, and prove the implementation is correct before trusting it.

**Load.** 6 objectives, ~5.5 hours.

### `prior_types` — Prior Types: Informative, Weakly Informative, and Flat

Distinguish informative, weakly informative, reference, and improper flat priors and say when each is appropriate.

- Bloom: **understand** · Difficulty: **2/5** · Est: **50 min**
- Prerequisites: `bayes_conjugacy`
- Assessment stems:
  1. Why is a Uniform(-1e6, 1e6) prior on a log-odds not 'uninformative'? Show its implied prior on the probability.
  1. Give a weakly informative prior for a regression slope on standardized data and justify the scale.
  1. Name a case where an improper prior yields an improper posterior.

### `prior_elicitation` — Prior Elicitation and Scale Setting

Turn domain knowledge and units into concrete prior distributions with defensible scales.

- Bloom: **create** · Difficulty: **3/5** · Est: **55 min**
- Prerequisites: `prior_types`
- Assessment stems:
  1. An expert says the effect is 'almost surely under 10 points.' Turn that into a prior and state the implied tail probability.
  1. Standardize predictors and set slope priors; explain how the priors change if you un-standardize.
  1. Elicit a prior for a positive scale parameter and defend Half-Normal vs Exponential vs Half-Cauchy.

### `prior_predictive_checks` — Prior Predictive Checks

Simulate data from the prior predictive distribution and reject priors that imply absurd data.

- Bloom: **evaluate** · Difficulty: **3/5** · Est: **55 min**
- Prerequisites: `prior_elicitation`, `prob_mc_integration`, `tool_python_prob_stack`
- Assessment stems:
  1. Simulate 200 prior predictive datasets for a logistic model and show the priors imply near-certain 0/1 outcomes.
  1. A height model's prior predictive contains negative heights. Diagnose which prior causes it and fix it.
  1. Describe a prior predictive check for a Poisson regression with a log link and an offset.

### `prior_sensitivity` — Prior Sensitivity Analysis

Quantify how much conclusions move when the prior is changed, and report it honestly.

- Bloom: **evaluate** · Difficulty: **3/5** · Est: **50 min**
- Prerequisites: `prior_predictive_checks`, `bayes_posterior_summaries`
- Assessment stems:
  1. Refit a model under three priors and tabulate how the key posterior quantity shifts.
  1. With n = 10,000, why does the prior usually stop mattering, and when does it still matter?
  1. Show a case where prior sensitivity is high precisely because the data are weakly identifying.

### `gen_model_writing` — Writing a Generative Model

Express a scientific question as a full generative model in statistical notation and in PPL code.

- Bloom: **create** · Difficulty: **3/5** · Est: **60 min**
- Prerequisites: `bayes_likelihood_function`, `prior_elicitation`, `tool_python_prob_stack`
- Assessment stems:
  1. Write the full generative model, in math and code, for test scores nested in schools.
  1. Given PPL code, back-translate it into complete statistical notation including all priors.
  1. Simulate a dataset from your own model, then recover the known parameters and report bias.

### `gen_sbc_fake_data` — Simulation-Based Calibration and Fake-Data Recovery

Validate a model implementation by recovering known parameters and checking rank uniformity.

- Bloom: **evaluate** · Difficulty: **4/5** · Est: **60 min**
- Prerequisites: `gen_model_writing`, `prior_predictive_checks`
- Assessment stems:
  1. Run simulation-based calibration on a hierarchical model and interpret the rank histogram.
  1. Your fake-data recovery is biased for one parameter only. List three candidate causes and how to test each.
  1. Why does fake-data recovery test the code and the model but not the world?


## Module 4: Computation: MCMC and Variational Inference

**Goal.** Understand and run the algorithms that make non-conjugate Bayesian inference possible.

**Load.** 8 objectives, ~7.7 hours.

### `mcmc_why_sampling` — Why Sampling: Intractable Posteriors and High Dimensions

Explain why normalizing constants are intractable and why grid and quadrature methods fail in high dimensions.

- Bloom: **understand** · Difficulty: **2/5** · Est: **45 min**
- Prerequisites: `bayes_conjugacy`, `prob_mc_integration`
- Assessment stems:
  1. Estimate the number of grid points needed for a 20-parameter model at 50 points per axis.
  1. Explain 'typical set' and why the posterior mode is not where the mass is in high dimensions.
  1. Name two models where conjugacy fails and say why.

### `mcmc_markov_chains` — Markov Chains, Stationarity, and Detailed Balance

Explain how a Markov chain can have a target posterior as its stationary distribution.

- Bloom: **understand** · Difficulty: **3/5** · Est: **55 min**
- Prerequisites: `mcmc_why_sampling`
- Assessment stems:
  1. Verify detailed balance for a simple two-state chain and find its stationary distribution.
  1. Why does MCMC output violate the independence assumption of plain Monte Carlo error formulas?
  1. State the conditions (irreducibility, aperiodicity) needed for convergence and give a chain that fails one.

### `mcmc_metropolis` — Metropolis-Hastings from Scratch

Implement Metropolis-Hastings, tune the proposal, and explain the acceptance ratio.

- Bloom: **apply** · Difficulty: **3/5** · Est: **60 min**
- Prerequisites: `mcmc_markov_chains`, `tool_python_prob_stack`
- Assessment stems:
  1. Code a random-walk Metropolis sampler for a Beta-Binomial posterior and compare to the closed form.
  1. Your acceptance rate is 2%. Diagnose the cause and fix it; then explain why 99% is also bad.
  1. Show why the proposal density cancels when the proposal is symmetric.

### `mcmc_hmc_nuts` — Hamiltonian Monte Carlo and NUTS

Explain gradient-based sampling, leapfrog integration, step size, and tree depth, and why HMC scales better.

- Bloom: **analyze** · Difficulty: **4/5** · Est: **70 min**
- Prerequisites: `mcmc_metropolis`, `prob_joint_marginal_transform`
- Assessment stems:
  1. Explain the roles of step size and number of leapfrog steps, and what happens when each is wrong.
  1. Why does HMC require a differentiable log density, and what does that forbid in your model?
  1. Contrast NUTS's automatic trajectory length with fixed-length HMC and state the trade-off.

### `tool_ppl_fitting` — Tooling: Fitting Models in a PPL (PyMC / Stan / NumPyro)

Specify, compile, and sample a model in at least one probabilistic programming language and read its output.

- Bloom: **apply** · Difficulty: **3/5** · Est: **70 min**
- Prerequisites: `gen_model_writing`, `mcmc_hmc_nuts`
- Assessment stems:
  1. Implement the same linear regression in two of PyMC, Stan, and NumPyro and confirm matching posteriors.
  1. Read a sampler warning about maximum tree depth and state the concrete next action.
  1. Explain what the PPL does with your prior statements at compile time.

### `tool_arviz_workflow` — Tooling: ArviZ and InferenceData for Diagnostics and Plots

Use ArviZ InferenceData to store, diagnose, compare, and visualize Bayesian model output.

- Bloom: **apply** · Difficulty: **2/5** · Est: **50 min**
- Prerequisites: `tool_ppl_fitting`
- Assessment stems:
  1. Build an InferenceData with posterior, prior, log_likelihood, and posterior_predictive groups.
  1. Produce a trace plot, rank plot, and forest plot for one model and say what each rules out.
  1. Use ArviZ to compare two models and read the output table correctly.

### `vi_basics` — Variational Inference: ELBO and Mean-Field

Explain VI as optimization of the ELBO and state what mean-field approximations distort.

- Bloom: **analyze** · Difficulty: **4/5** · Est: **60 min**
- Prerequisites: `mcmc_why_sampling`, `prob_joint_marginal_transform`
- Assessment stems:
  1. Derive the ELBO from the KL divergence and say which direction of KL is being minimized.
  1. Explain why mean-field VI typically underestimates posterior variance; illustrate on a correlated Normal.
  1. Given a 2M-row dataset with a deadline, argue for or against VI over NUTS.

### `vi_advi_practice` — ADVI and Practical Variational Workflows

Run automatic differentiation VI, diagnose its failure modes, and decide when to fall back to MCMC.

- Bloom: **apply** · Difficulty: **4/5** · Est: **50 min**
- Prerequisites: `vi_basics`, `tool_ppl_fitting`
- Assessment stems:
  1. Fit a model with ADVI and with NUTS and compare posterior means and sds; report where they diverge.
  1. Interpret a non-converging ELBO trace and list two fixes.
  1. Describe a diagnostic (e.g. importance-weighted check) that detects a bad variational approximation.


## Module 5: Diagnostics and Posterior Predictive Checking

**Goal.** Decide whether a fit is trustworthy, and whether the model can reproduce the features of the data you care about.

**Load.** 7 objectives, ~5.8 hours.

### `diag_trace_convergence` — Trace Plots, Warmup, and Convergence by Eye

Read trace and rank plots to spot non-convergence, stuck chains, and insufficient warmup.

- Bloom: **analyze** · Difficulty: **2/5** · Est: **45 min**
- Prerequisites: `tool_arviz_workflow`, `mcmc_markov_chains`
- Assessment stems:
  1. Given four trace plots, rank them from healthy to broken and justify each call.
  1. Why must warmup draws be discarded, and why is warmup not just 'burn-in'?
  1. Two chains explore different modes. What does that say about the posterior and the model?

### `diag_rhat` — R-hat and Between-Chain Agreement

Compute and interpret split R-hat and explain what it can and cannot detect.

- Bloom: **analyze** · Difficulty: **3/5** · Est: **45 min**
- Prerequisites: `diag_trace_convergence`
- Assessment stems:
  1. All R-hat values are 1.00 but the model is wrong. Explain why R-hat did not catch it.
  1. Why does R-hat require multiple chains with dispersed inits?
  1. Interpret R-hat = 1.08 for one parameter and give the next three actions.

### `diag_ess_mcse` — Effective Sample Size and Monte Carlo Standard Error

Use bulk and tail ESS and MCSE to decide whether enough draws have been collected.

- Bloom: **evaluate** · Difficulty: **3/5** · Est: **50 min**
- Prerequisites: `diag_rhat`, `prob_mc_integration`
- Assessment stems:
  1. You have 4000 draws and bulk ESS 120. How many more draws are needed for MCSE below target?
  1. Explain the difference between bulk ESS and tail ESS and which matters for a 99th percentile forecast.
  1. Justify reporting posterior means to only two significant digits using MCSE.

### `diag_divergences` — Divergences, Tree Depth, and Energy Diagnostics

Diagnose HMC-specific pathologies and take the right corrective action.

- Bloom: **evaluate** · Difficulty: **4/5** · Est: **55 min**
- Prerequisites: `mcmc_hmc_nuts`, `diag_ess_mcse`
- Assessment stems:
  1. You get 47 divergences concentrated in one region. Describe the geometry and two fixes.
  1. Interpret a BFMI warning and say what it implies about the model, not just the sampler.
  1. Why is 'raise target_accept until divergences vanish' sometimes hiding a real problem?

### `ppc_posterior_predictive` — The Posterior Predictive Distribution

Define and simulate the posterior predictive distribution and distinguish it from the posterior of a parameter.

- Bloom: **apply** · Difficulty: **3/5** · Est: **50 min**
- Prerequisites: `bayes_posterior_summaries`, `prob_mc_integration`, `tool_arviz_workflow`
- Assessment stems:
  1. Simulate posterior predictive draws for a Poisson model and explain why they are wider than the posterior mean's spread.
  1. State the integral that defines the posterior predictive and what is being marginalized out.
  1. A stakeholder wants 'the interval for the next observation.' Which distribution do you use and why?

### `ppc_checks` — Posterior Predictive Checking and Test Quantities

Design test quantities that would expose the specific failures that matter, and read PPC plots.

- Bloom: **evaluate** · Difficulty: **3/5** · Est: **55 min**
- Prerequisites: `ppc_posterior_predictive`
- Assessment stems:
  1. Your model fits the mean but not the count of zeros. Choose a test quantity and show the misfit.
  1. Explain a Bayesian p-value for a test quantity and why values near 0.5 are not proof of a good model.
  1. Design three test quantities for a sales forecasting model, one per failure mode you fear.

### `ppc_residuals_misfit` — Residual Analysis and Locating Misfit

Use standardized and randomized-quantile residuals to find where and for whom a model fails.

- Bloom: **analyze** · Difficulty: **3/5** · Est: **50 min**
- Prerequisites: `ppc_checks`
- Assessment stems:
  1. Compute randomized quantile residuals for a negative binomial model and check uniformity.
  1. Group residuals by a held-out covariate to reveal a missing interaction.
  1. Why can raw residuals mislead for discrete or heteroskedastic outcomes?


## Module 6: Bayesian Regression: GLMs, Robustness, and Smooths

**Goal.** Model a conditional mean honestly across outcome types, nonlinearity, outliers, and imperfect data.

**Load.** 6 objectives, ~5.5 hours.

### `reg_linear_bayesian` — Bayesian Linear Regression

Fit and interpret a Bayesian linear regression, including priors on coefficients and the residual scale.

- Bloom: **apply** · Difficulty: **3/5** · Est: **60 min**
- Prerequisites: `gen_model_writing`, `tool_ppl_fitting`, `prior_elicitation`
- Assessment stems:
  1. Fit a regression with standardized predictors, report coefficient posteriors, and interpret one in original units.
  1. Explain what the posterior of sigma tells you that R-squared does not.
  1. Add a redundant, highly collinear predictor and describe what happens to the coefficient posteriors.

### `reg_glm_links` — Generalized Linear Models and Link Functions

Build logistic, Poisson, and negative binomial regressions and interpret parameters on the correct scale.

- Bloom: **apply** · Difficulty: **3/5** · Est: **65 min**
- Prerequisites: `reg_linear_bayesian`, `prob_common_distributions`
- Assessment stems:
  1. Interpret a logistic coefficient of 0.7 as an odds ratio and as a probability change at two baselines.
  1. Add an exposure offset to a Poisson model and explain why it is not a coefficient.
  1. Data are overdispersed counts. Move from Poisson to negative binomial and show the effect on intervals.

### `reg_robust_outliers` — Robust Regression and Heavy Tails

Replace Normal likelihoods with Student-t or mixture likelihoods to limit outlier influence.

- Bloom: **apply** · Difficulty: **3/5** · Est: **45 min**
- Prerequisites: `reg_linear_bayesian`, `prob_common_distributions`
- Assessment stems:
  1. Refit a regression with a Student-t likelihood and compare slope posteriors to the Normal fit.
  1. Explain why estimating nu rather than fixing it is often better, and what the posterior of nu tells you.
  1. Contrast deleting outliers with modeling them; state the inferential cost of each.

### `reg_interactions_nonlinearity` — Interactions, Polynomials, and Nonlinear Effects

Model nonlinearity and effect modification, and interpret them via marginal effects.

- Bloom: **analyze** · Difficulty: **3/5** · Est: **50 min**
- Prerequisites: `reg_glm_links`
- Assessment stems:
  1. Interpret an interaction term in a logistic model at three covariate values using predicted probabilities.
  1. Explain why a polynomial fit extrapolates dangerously, and show it on held-out range.
  1. Compute and plot an average marginal effect from posterior draws.

### `reg_splines_gams` — Splines and Gaussian-Additive Smooths

Use basis expansions and penalized splines to fit smooth nonlinear relationships with controlled wiggliness.

- Bloom: **create** · Difficulty: **4/5** · Est: **55 min**
- Prerequisites: `reg_interactions_nonlinearity`, `prior_elicitation`
- Assessment stems:
  1. Fit a P-spline to a nonlinear trend and show how the smoothing prior controls wiggliness.
  1. Explain the role of knot count versus the penalty prior; which matters more and why?
  1. Compare a spline fit to a Gaussian process fit on the same data and state the trade-offs.

### `reg_measurement_missing` — Measurement Error and Missing Data

Model measurement error and missingness generatively instead of deleting or imputing naively.

- Bloom: **create** · Difficulty: **4/5** · Est: **55 min**
- Prerequisites: `reg_linear_bayesian`, `gen_model_writing`
- Assessment stems:
  1. Write a model where a predictor is observed with known error and show the effect on the slope posterior.
  1. Contrast MCAR, MAR, and MNAR and say which one complete-case analysis needs.
  1. Explain why Bayesian imputation of a missing outcome is just marginalization.


## Module 7: Hierarchical and Multilevel Models

**Goal.** Share strength across groups, and fix the sampling geometry that hierarchical models create.

**Load.** 6 objectives, ~5.7 hours.

### `hier_partial_pooling` — Partial Pooling: No Pooling, Complete Pooling, and Between

Explain hierarchical shrinkage as a compromise between pooled and separate estimates.

- Bloom: **analyze** · Difficulty: **3/5** · Est: **60 min**
- Prerequisites: `reg_linear_bayesian`, `bayes_sequential_updating`
- Assessment stems:
  1. Fit all three pooling strategies to grouped data and plot the shrinkage of small-group estimates.
  1. Why do small groups shrink more than large ones? Show the algebra for the Normal-Normal case.
  1. A manager says shrinkage 'hides real differences.' Give the bias-variance answer.

### `mcmc_reparameterization` — Reparameterization: Centered vs Non-Centered

Recognize funnel geometry and reparameterize a model to make the posterior easier to sample.

- Bloom: **create** · Difficulty: **4/5** · Est: **60 min**
- Prerequisites: `mcmc_hmc_nuts`, `hier_partial_pooling`
- Assessment stems:
  1. Rewrite a centered hierarchical model in non-centered form and show the divergence count drop.
  1. Plot the funnel for a small group-count model and explain why small tau causes divergences.
  1. When is the centered parameterization actually the better choice?

### `diag_workflow_triage` — Diagnostic Triage: A Decision Procedure

Apply an ordered checklist that turns sampler warnings into concrete model or code changes.

- Bloom: **create** · Difficulty: **4/5** · Est: **50 min**
- Prerequisites: `diag_divergences`, `mcmc_reparameterization`, `gen_sbc_fake_data`
- Assessment stems:
  1. Write your own triage flowchart from warnings to actions and defend the ordering.
  1. A model has high R-hat, low ESS, and divergences at once. What do you fix first and why?
  1. Distinguish a computation problem from a model problem given only diagnostic output.

### `hier_varying_intercepts_slopes` — Varying Intercepts and Varying Slopes

Build multilevel models with group-varying intercepts and slopes and a correlation structure between them.

- Bloom: **create** · Difficulty: **4/5** · Est: **65 min**
- Prerequisites: `hier_partial_pooling`, `tool_ppl_fitting`
- Assessment stems:
  1. Fit a varying-intercept, varying-slope model and interpret the intercept-slope correlation posterior.
  1. Choose an LKJ prior for the correlation matrix and justify the eta value.
  1. Explain what the group-level sd posterior being near zero implies.

### `hier_group_predictors` — Group-Level Predictors and Cross-Level Interactions

Add covariates at the group level and interpret cross-level interactions correctly.

- Bloom: **analyze** · Difficulty: **4/5** · Est: **50 min**
- Prerequisites: `hier_varying_intercepts_slopes`
- Assessment stems:
  1. Add a school-level funding predictor to a student-level model and interpret its coefficient.
  1. Explain why a group-level predictor reduces the group-level sd rather than the residual sd.
  1. Distinguish within-group and between-group effects and show how to separate them by centering.

### `hier_nonnested_multilevel` — Non-Nested, Crossed, and Deep Hierarchies

Model crossed and multi-way grouping structures and know their computational cost.

- Bloom: **create** · Difficulty: **5/5** · Est: **55 min**
- Prerequisites: `hier_group_predictors`, `diag_workflow_triage`
- Assessment stems:
  1. Specify a model with crossed product and region effects and describe the resulting design.
  1. Why do deep hierarchies with few groups per level produce weakly identified variance components?
  1. Give a rule of thumb for the minimum number of groups before a hierarchical level is worth it.


## Module 8: Model Comparison and Averaging

**Goal.** Compare models on expected out-of-sample predictive accuracy, and combine them rather than over-trusting one.

**Load.** 4 objectives, ~3.5 hours.

### `cmp_predictive_view` — Predictive Accuracy as the Comparison Target

Frame model comparison as expected out-of-sample predictive accuracy rather than in-sample fit.

- Bloom: **understand** · Difficulty: **3/5** · Est: **45 min**
- Prerequisites: `ppc_checks`
- Assessment stems:
  1. Explain overfitting using expected log predictive density rather than 'too many parameters'.
  1. Why does adding a predictor always improve in-sample fit but not necessarily prediction?
  1. Define elpd in words and say what quantity it is an expectation over.

### `cmp_waic_loo` — WAIC and PSIS-LOO Cross-Validation

Compute WAIC and PSIS-LOO, interpret elpd differences with standard errors, and check Pareto k values.

- Bloom: **evaluate** · Difficulty: **4/5** · Est: **60 min**
- Prerequisites: `cmp_predictive_view`, `tool_arviz_workflow`
- Assessment stems:
  1. Two models differ by elpd_diff = 3.1 with se_diff = 4.0. What do you conclude?
  1. Several Pareto k values exceed 0.7. Explain what that means and give two remedies.
  1. Why is LOO invalid as written for time series, and what replaces it?

### `cmp_bayes_factors` — Marginal Likelihood, Bayes Factors, and Their Hazards

Compute and interpret Bayes factors while recognizing their extreme prior sensitivity.

- Bloom: **evaluate** · Difficulty: **4/5** · Est: **55 min**
- Prerequisites: `prior_sensitivity`, `cmp_waic_loo`
- Assessment stems:
  1. Show numerically how widening a prior by 10x changes a Bayes factor but barely changes the posterior.
  1. Explain the Jeffreys-Lindley paradox in one paragraph with a concrete example.
  1. When is a Bayes factor the right tool, and when is LOO the right tool?

### `cmp_stacking_averaging` — Model Averaging, Stacking, and Ensembles

Combine models by stacking or pseudo-BMA weights instead of picking a single winner.

- Bloom: **create** · Difficulty: **4/5** · Est: **50 min**
- Prerequisites: `cmp_waic_loo`
- Assessment stems:
  1. Compute stacking weights for three candidate forecasting models and interpret a zero weight.
  1. Contrast stacking with Bayesian model averaging under an open-world assumption.
  1. Show a case where an averaged forecast beats every individual model on log score.


## Module 9: Bayesian Time Series and Structural Forecasting

**Goal.** Model time-dependent data with state space, seasonal, GP, and hierarchical structure, and produce honest predictive distributions.

**Load.** 10 objectives, ~9.6 hours.

### `ts_foundations` — Time Series Foundations: Stationarity, ACF, and Decomposition

Diagnose trend, seasonality, and autocorrelation, and explain stationarity and differencing.

- Bloom: **analyze** · Difficulty: **2/5** · Est: **55 min**
- Prerequisites: `prob_expectation_variance`, `reg_linear_bayesian`
- Assessment stems:
  1. Given an ACF and PACF plot, propose an AR or MA order and justify it.
  1. Explain why a random walk is non-stationary and what differencing does to its variance.
  1. Decompose a series into trend, seasonal, and remainder and state what the remainder should look like.

### `ts_ar_arima` — Bayesian AR, MA, and ARIMA Models

Fit autoregressive and ARIMA-style models in a Bayesian framework with priors that enforce stability.

- Bloom: **apply** · Difficulty: **3/5** · Est: **60 min**
- Prerequisites: `ts_foundations`, `tool_ppl_fitting`
- Assessment stems:
  1. Fit an AR(2) with priors keeping the roots outside the unit circle and interpret the coefficients.
  1. Explain how the initial conditions are handled and why they matter for short series.
  1. Show the posterior forecast fan from an ARIMA model and explain why it widens.

### `ts_local_level_trend` — State Space Models: Local Level and Local Linear Trend

Express a series as a latent state plus observation noise and fit local level and local trend models.

- Bloom: **create** · Difficulty: **4/5** · Est: **65 min**
- Prerequisites: `ts_ar_arima`, `reg_measurement_missing`
- Assessment stems:
  1. Fit a local level model and interpret the ratio of state noise to observation noise.
  1. Show how a local linear trend model handles a level shift, and where it fails.
  1. Explain how missing observations are handled naturally in a state space formulation.

### `ts_seasonality` — Seasonality: Dummy, Fourier, and Time-Varying Seasonal Effects

Represent single and multiple seasonalities and allow seasonal patterns to evolve.

- Bloom: **apply** · Difficulty: **3/5** · Est: **55 min**
- Prerequisites: `ts_local_level_trend`
- Assessment stems:
  1. Model weekly and yearly seasonality with Fourier terms and choose the number of harmonics.
  1. Contrast a fixed seasonal dummy set with a time-varying seasonal state; when is each right?
  1. Handle a series with holidays that move each year, and state the modeling choice.

### `ts_structural_bsts` — Structural Time Series and BSTS-Style Decomposition

Compose trend, seasonal, cycle, and regression components into an interpretable structural model.

- Bloom: **create** · Difficulty: **4/5** · Est: **60 min**
- Prerequisites: `ts_seasonality`, `reg_glm_links`
- Assessment stems:
  1. Build a structural model with trend, weekly seasonality, and a promotion regressor, then plot the component decomposition.
  1. Explain how spike-and-slab or regularized priors handle many candidate regressors.
  1. Attribute a forecast change to a specific component and defend the attribution.

### `ts_gaussian_processes` — Gaussian Processes for Time Series

Use GP priors with interpretable kernels to model smooth trends, periodicity, and uncertainty growth.

- Bloom: **create** · Difficulty: **5/5** · Est: **70 min**
- Prerequisites: `reg_splines_gams`, `ts_seasonality`
- Assessment stems:
  1. Compose a kernel for a smooth trend plus yearly periodicity and describe each hyperparameter's role.
  1. Explain why a GP's forecast uncertainty reverts toward the prior far from data.
  1. Given O(n^3) cost, name two approximations and their assumptions.

### `ts_intermittent_counts` — Intermittent Demand, Counts, and Zero Inflation

Forecast sparse, zero-heavy, or count-valued series with appropriate likelihoods.

- Bloom: **apply** · Difficulty: **4/5** · Est: **45 min**
- Prerequisites: `ts_structural_bsts`, `reg_glm_links`
- Assessment stems:
  1. Model a series that is zero on 70% of days and justify the likelihood choice.
  1. Explain why a Normal forecast interval is wrong for low counts, with a numeric example.
  1. Distinguish structural zeros from sampling zeros in a demand series.

### `ts_changepoints_regimes` — Changepoints, Regime Shifts, and Structural Breaks

Detect and model level shifts, trend breaks, and regime switching in a Bayesian framework.

- Bloom: **create** · Difficulty: **5/5** · Est: **55 min**
- Prerequisites: `ts_local_level_trend`, `ppc_residuals_misfit`
- Assessment stems:
  1. Fit a model with an unknown changepoint location and report the posterior over the changepoint.
  1. Contrast an explicit changepoint model with a state space model whose state noise is large.
  1. Explain the forecasting risk of fitting a break that is really noise.

### `ts_hierarchical_forecasting` — Hierarchical and Grouped Forecasting with Reconciliation

Forecast many related series with shared structure and reconcile forecasts across an aggregation hierarchy.

- Bloom: **create** · Difficulty: **5/5** · Est: **60 min**
- Prerequisites: `hier_varying_intercepts_slopes`, `ts_structural_bsts`
- Assessment stems:
  1. Forecast store-level sales and reconcile them to regional and national totals; explain the coherence constraint.
  1. Show how partial pooling across series improves forecasts for short-history series.
  1. Contrast bottom-up, top-down, and optimal reconciliation and state when each wins.

### `ts_forecast_uncertainty` — Forecast Distributions and Uncertainty Propagation

Produce multi-step-ahead predictive distributions that propagate parameter and process uncertainty.

- Bloom: **analyze** · Difficulty: **4/5** · Est: **50 min**
- Prerequisites: `ppc_posterior_predictive`, `ts_ar_arima`
- Assessment stems:
  1. Generate 12-step-ahead predictive draws and explain each source contributing to the widening fan.
  1. Show why plugging in posterior mean parameters understates forecast uncertainty.
  1. Compute the predictive distribution of a cumulative 12-month total, not just each month.


## Module 10: Forecast Evaluation, Decisions, and the Full Workflow

**Goal.** Prove a forecast is good with proper scores and calibration, turn it into decisions, and run the whole workflow end to end.

**Load.** 9 objectives, ~10.6 hours.

### `ts_backtesting` — Backtesting: Rolling Origin and Time Series Cross-Validation

Evaluate forecasts with rolling-origin or expanding-window schemes that respect time order.

- Bloom: **evaluate** · Difficulty: **4/5** · Est: **55 min**
- Prerequisites: `ts_forecast_uncertainty`, `cmp_waic_loo`
- Assessment stems:
  1. Design a rolling-origin backtest for a daily series with weekly seasonality and a 14-day horizon.
  1. Explain a concrete leakage path in a naive k-fold split of a time series.
  1. Report scores by horizon and explain why an average over horizons can mislead.

### `ts_scoring_rules` — Proper Scoring Rules: Log Score, CRPS, and Interval Scores

Score probabilistic forecasts with proper rules and explain why impropriety invites gaming.

- Bloom: **evaluate** · Difficulty: **4/5** · Est: **60 min**
- Prerequisites: `ts_backtesting`, `cmp_predictive_view`
- Assessment stems:
  1. Compute CRPS for a sample-based predictive distribution and compare to MAE of the median.
  1. Show that a scoring rule like MAPE is improper or degenerate, and construct a forecast that games it.
  1. A model wins on CRPS but loses on log score. Explain what that reveals about its tails.

### `ts_calibration_sharpness` — Forecast Calibration and Sharpness

Assess calibration with PIT histograms and coverage, and balance it against sharpness.

- Bloom: **evaluate** · Difficulty: **4/5** · Est: **50 min**
- Prerequisites: `ts_scoring_rules`
- Assessment stems:
  1. Plot a PIT histogram for 500 forecasts; a U-shape appears. Diagnose the cause.
  1. State the 'sharpness subject to calibration' principle and give a forecast that is calibrated but useless.
  1. Report empirical coverage of 50% and 90% intervals by horizon and interpret the pattern.

### `dec_loss_functions` — Decision Theory: Loss Functions and Optimal Actions

Choose actions that minimize expected posterior loss instead of reporting point estimates.

- Bloom: **create** · Difficulty: **4/5** · Est: **55 min**
- Prerequisites: `bayes_posterior_summaries`, `ppc_posterior_predictive`
- Assessment stems:
  1. Under asymmetric loss (stockouts cost 5x overstock), derive the optimal order quantity from predictive draws.
  1. Show that the posterior mean, median, and mode are optimal under squared, absolute, and 0-1 loss.
  1. A threshold decision uses P(effect > 0) > 0.95. Restate the implied loss function.

### `dec_value_of_information` — Value of Information and Deciding What to Measure

Estimate how much a decision would improve with more data, and decide whether to collect it.

- Bloom: **evaluate** · Difficulty: **5/5** · Est: **45 min**
- Prerequisites: `dec_loss_functions`
- Assessment stems:
  1. Compute the expected value of perfect information for a two-option decision from posterior draws.
  1. Design a Bayesian sample-size or stopping analysis based on decision quality, not power.
  1. Argue why more data is sometimes not worth its cost, with numbers.

### `dec_causal_cautions` — Causal Cautions: Confounding, Colliders, and Bayesian Humility

Recognize that Bayesian machinery does not make an observational estimate causal, and use DAG reasoning to pick adjustments.

- Bloom: **analyze** · Difficulty: **4/5** · Est: **60 min**
- Prerequisites: `reg_interactions_nonlinearity`, `hier_group_predictors`
- Assessment stems:
  1. Draw a DAG where adjusting for a collider creates bias, and show the effect in simulation.
  1. A stakeholder reads a posterior slope as 'the effect of price on sales.' List the assumptions required.
  1. Explain why adding every available covariate is not a safe default.

### `dec_communication` — Communicating Uncertainty to Decision Makers

Present posterior and predictive uncertainty in ways non-statisticians can act on.

- Bloom: **create** · Difficulty: **3/5** · Est: **40 min**
- Prerequisites: `bayes_posterior_summaries`, `ts_calibration_sharpness`
- Assessment stems:
  1. Rewrite a fan-chart forecast as three sentences an operations manager can act on.
  1. Choose between a fan chart, a set of scenario draws, and a probability table for one audience; defend it.
  1. Identify three ways a 95% interval is routinely misread and how your visual prevents each.

### `workflow_bayesian_end_to_end` — The Bayesian Workflow End to End

Run the full loop: model, prior check, fit, diagnose, posterior check, compare, expand, decide, document.

- Bloom: **create** · Difficulty: **5/5** · Est: **90 min**
- Prerequisites: `diag_workflow_triage`, `ppc_residuals_misfit`, `cmp_stacking_averaging`, `prior_sensitivity`, `dec_communication`
- Assessment stems:
  1. Take a raw dataset to a documented decision, showing every workflow stage and each decision you made.
  1. Given a colleague's notebook, audit which workflow stages are missing and what risk each omission creates.
  1. Describe how you would version models and results so a fit from six months ago is reproducible.

### `forecast_capstone` — Capstone: Production-Grade Probabilistic Forecast

Deliver a backtested, calibrated, hierarchical forecasting system with a decision layer and monitoring.

- Bloom: **create** · Difficulty: **5/5** · Est: **180 min**
- Prerequisites: `ts_hierarchical_forecasting`, `ts_calibration_sharpness`, `ts_changepoints_regimes`, `ts_intermittent_counts`, `dec_value_of_information`, `workflow_bayesian_end_to_end`
- Assessment stems:
  1. Build, backtest, and calibrate a multi-series forecast, then defend every modeling choice against a named alternative.
  1. Define a monitoring plan that detects forecast degradation before the business does.
  1. Present the system to a skeptical executive: what you know, what you do not, and what you would do next.


## Tracked misconceptions

The Assessor tags these when a wrong answer matches the pattern. The Author then writes
targeted remediation, because a student holding one of these will keep producing correct-
looking work for the wrong reason.

### `mis_posterior_prob_of_data`

**Wrong model:** The p-value and the posterior probability of the hypothesis are the same number; P(data|H) can be read as P(H|data).

**Attaches to:** `bayes_theorem`, `bayes_likelihood_function`, `bayes_credible_vs_confidence`

**Diagnostic:** A test is positive. Sensitivity is 99%. Prevalence is 0.1%. A student says 'so there is a 99% chance the patient is sick.' What is the actual probability, and which conditional did the student flip?

### `mis_flat_prior_is_neutral`

**Wrong model:** A flat or very wide prior is 'objective' and adds no information.

**Attaches to:** `prior_types`, `prior_elicitation`, `prior_predictive_checks`

**Diagnostic:** You put Uniform(-50, 50) on a logistic regression coefficient. Simulate the prior predictive. What proportion of simulated outcome probabilities land within 0.05 of 0 or 1, and is that a neutral belief?

### `mis_prior_swamped_always`

**Wrong model:** With enough data the prior never matters, so prior choice is never worth arguing about.

**Attaches to:** `prior_sensitivity`, `hier_partial_pooling`, `cmp_bayes_factors`

**Diagnostic:** You have 500,000 rows but 8 groups at the top hierarchical level, and a near-separated logistic term. Name two parameters where the prior still drives the answer, and explain why n did not help.

### `mis_more_draws_fixes_bias`

**Wrong model:** If the sampler shows problems, running more iterations will fix it.

**Attaches to:** `diag_rhat`, `diag_ess_mcse`, `diag_divergences`, `diag_workflow_triage`

**Diagnostic:** A run has R-hat 1.4 and 300 divergences. You rerun with 10x the draws and R-hat is still 1.35. What class of problem is this, and what is the correct next action?

### `mis_rhat_means_correct`

**Wrong model:** R-hat near 1.00 and high ESS mean the model is correct.

**Attaches to:** `diag_rhat`, `ppc_checks`, `cmp_predictive_view`

**Diagnostic:** A Normal model fitted to strictly positive, heavily skewed counts converges perfectly, with R-hat 1.00 everywhere. What does convergence actually certify, and which check would expose the misfit?

### `mis_divergences_ignorable`

**Wrong model:** A few divergences are a cosmetic sampler warning and can be ignored if the trace looks fine.

**Attaches to:** `diag_divergences`, `mcmc_reparameterization`

**Diagnostic:** Your hierarchical model has 12 divergences, all with tau below 0.1. Plot tau against the group effects. What region of the posterior is being missed, and how does that bias your group-level sd estimate?

### `mis_posterior_vs_predictive`

**Wrong model:** The credible interval for the mean is also the interval in which the next observation will fall.

**Attaches to:** `ppc_posterior_predictive`, `ts_forecast_uncertainty`, `dec_communication`

**Diagnostic:** Your posterior 95% interval for the mean daily sales is [98, 102]. What fraction of actual days fall in that interval, and which distribution should you have quoted?

### `mis_plugin_forecast`

**Wrong model:** Forecasting means plugging the posterior mean parameters into the model and projecting forward.

**Attaches to:** `ts_forecast_uncertainty`, `ppc_posterior_predictive`, `ts_calibration_sharpness`

**Diagnostic:** Compare 12-step forecast intervals from (a) plug-in posterior means and (b) full predictive draws. Which is narrower, by roughly how much, and which source of uncertainty was dropped?

### `mis_loo_for_timeseries`

**Wrong model:** LOO or k-fold cross-validation can be applied to time series the same way as to i.i.d. data.

**Attaches to:** `cmp_waic_loo`, `ts_backtesting`

**Diagnostic:** You run standard PSIS-LOO on a daily series with strong autocorrelation and it prefers the most flexible model. Explain the leakage, and design the evaluation scheme that replaces it.

### `mis_bayes_factor_equals_loo`

**Wrong model:** Bayes factors and LOO answer the same question, so either can be used interchangeably.

**Attaches to:** `cmp_bayes_factors`, `cmp_waic_loo`, `prior_sensitivity`

**Diagnostic:** Widen the prior on the tested coefficient by a factor of 10. Report the change in the Bayes factor and the change in elpd_loo. Why does one move a lot and the other barely at all?

### `mis_shrinkage_is_cheating`

**Wrong model:** Partial pooling distorts the data by pulling group estimates toward the mean, so no-pooling estimates are more honest.

**Attaches to:** `hier_partial_pooling`, `hier_varying_intercepts_slopes`

**Diagnostic:** One store has 3 observations and a no-pooling mean of 0.95; the global mean is 0.40. Which estimate has lower expected out-of-sample error next quarter, and why is 'honesty' the wrong criterion here?

### `mis_accuracy_metric_is_enough`

**Wrong model:** A good forecast is one with low MAE or MAPE; the interval is decoration.

**Attaches to:** `ts_scoring_rules`, `ts_calibration_sharpness`, `dec_loss_functions`

**Diagnostic:** Two forecasters have identical MAE. One reports 95% intervals with 60% empirical coverage. Rank them by CRPS and log score, and explain which business decision changes.

### `mis_regression_is_causal`

**Wrong model:** A coefficient with a posterior far from zero shows that the predictor causes the outcome, especially in a Bayesian model.

**Attaches to:** `dec_causal_cautions`, `reg_interactions_nonlinearity`, `hier_group_predictors`

**Diagnostic:** Price has a positive posterior slope on sales, clearly excluding zero. List the causal assumptions needed to act on it, and name one variable whose omission (or whose inclusion as a collider) would flip the sign.

### `mis_control_for_everything`

**Wrong model:** Adding more covariates always reduces bias, so control for everything available.

**Attaches to:** `dec_causal_cautions`, `reg_measurement_missing`

**Diagnostic:** Simulate X -> C <- Y and regress Y on X adjusting for C. Report the estimated X coefficient with and without C, given that the true effect is zero. Why did adjustment create the bias?

### `mis_vi_equals_mcmc`

**Wrong model:** Variational inference is just faster MCMC and gives the same posterior.

**Attaches to:** `vi_basics`, `vi_advi_practice`

**Diagnostic:** Fit a correlated 2D posterior with mean-field ADVI and with NUTS. Compare the posterior sds and the correlation. Which is systematically wrong, in which direction, and why does the KL direction cause it?


## Deliberately excluded

Kept out to protect the critical path to forecasting. Each is a candidate for a later
extension graph, not a gap in the core.

- **Nonparametric Bayes** — Dirichlet processes, Indian buffet, infinite mixtures.
- **Bayesian neural networks and deep probabilistic models**, including normalizing flows
  and deep state space models.
- **Sequential Monte Carlo / particle filters**, and online filtering for non-linear state
  space models. Kalman-style state space is covered; particle methods are not.
- **Simulation-based inference / ABC** for likelihood-free models.
- **Bayesian optimization, bandits, and reinforcement learning.**
- **Bayesian experimental design and adaptive clinical trials** beyond the value-of-information
  objective.
- **Full causal inference** — do-calculus, instrumental variables, difference-in-differences,
  synthetic control. The graph teaches causal *caution* only: enough to stop a student calling
  a coefficient an effect.
- **Spatial and spatio-temporal models** (CAR, ICAR, spatial GPs).
- **Bayesian survival analysis and point processes.**
- **Gibbs sampling and conditional conjugacy derivations.** The requested computation strand is
  Metropolis plus HMC/NUTS plus VI; modern PPL practice rarely needs hand-derived Gibbs.
- **Production engineering** — feature stores, orchestration, serving. The capstone asks for a
  monitoring *plan*, not an MLOps build.

## Tooling stance

The graph is PPL-agnostic. `tool_ppl_fitting` explicitly asks the student to implement one
model in two of PyMC, Stan, and NumPyro, so the student learns the model, not the API.
ArviZ is the shared diagnostic and comparison layer across all three, and
`tool_arviz_workflow` is a prerequisite for every diagnostic and comparison objective.
