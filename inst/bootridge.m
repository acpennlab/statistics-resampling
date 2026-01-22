% Empirical Bayes Ridge Regression with ridge tuning via .632 bootstrap
% prediction error for univariate or multivariate outcomes.
%
% -- Function File: bootridge (Y, X)
% -- Function File: bootridge (Y, X, CATEGOR)
% -- Function File: bootridge (Y, X, CATEGOR, NBOOT)
% -- Function File: bootridge (Y, X, CATEGOR, NBOOT, ALPHA)
% -- Function File: bootridge (Y, X, CATEGOR, NBOOT, ALPHA, L)
% -- Function File: bootridge (Y, X, CATEGOR, NBOOT, ALPHA, L, DEFF)
% -- Function File: S = bootridge (Y, X, CATEGOR, NBOOT, ALPHA, L, DEFF, SEED)
%
%      'bootridge (Y, X)' fits an empirical Bayes ridge regression model using
%      a linear Normal (Gaussian) likelihood with an empirical Bayes normal
%      ridge prior on the regression coefficients. The ridge tuning constant
%      (lambda) is selected by minimizing the .632 bootstrap estimate of
%      prediction error calculated using the mean Huber loss [1, 2]. Y is an 
%      m-by-q matrix of outcomes and X is an m-by-n design matrix whose first 
%      column must correspond to an intercept term. Missing or infinite values 
%      in X or Y are omitted before fitting.
%
%      For each outcome, the function prints posterior summaries including 
%      means, equal-tailed credible intervals, and Bayes factors (lnBF10). 
%      When q > 1, summaries for residual correlations between outcomes are 
%      additionally provided.
%
%      Interpretation note (empirical Bayes):
%        Bayes factors reported by 'bootridge' are empirical‑Bayes approximations
%        based on a data‑tuned ridge prior. They are best viewed as model‑
%        comparison diagnostics (evidence on a predictive scale) rather than 
%        literal posterior odds [3–5]. The log scale (lnBF10) is numerically 
%        stable and recommended for interpretation.
%
%      'bootridge (Y, X, CATEGOR)' selects predictor columns corresponding to
%      categorical variables (scalar or vector of column indices, excluding
%      the intercept). CATEGOR ensures that variance‑based penalty scaling is 
%      not applied to coded categorical terms (categoricals use a unit penalty)
%
%      'bootridge (Y, X, CATEGOR, NBOOT)' sets the number of bootstrap samples 
%      used for tuning (default = 100). The function utilizes balanced 
%      bootknife (leave-one-out) resampling to ensure valid out-of-bag samples.
%
%      'bootridge (..., ALPHA)' sets the two‑tailed error rate for credibility
%      intervals (default ALPHA = 0.05 for 95% central credible intervals).
%
%      'bootridge (Y, X, CATEGOR, NBOOT, ALPHA, L)' specifies a hypothesis 
%      matrix L (n-by-c) to report summaries for linear contrasts or 
%      estimates rather than raw model coefficients.
%
%      'bootridge (Y, X, CATEGOR, NBOOT, ALPHA, L, DEFF)' specifies a design 
%      effect to account for clustering. DEFF inflates the posterior 
%      covariance and reduces the effective degrees of freedom (df_t) to 
%      calibrate inference for effective sample size.
%
%      'bootridge (Y, X, CATEGOR, NBOOT, ALPHA, L, DEFF, SEED)' sets the 
%      bootstrap resampling seed for reproducible and stable tuning.
%
%      'S = bootridge (Y, X, ...)' returns a structure containing posterior 
%      summaries, the optimized ridge parameter (lambda), residual 
%      covariance (Sigma_Y_hat), and optimization diagnostics.
%
%      DETAIL: The model implements an empirical Bayes ridge regression that
%      simultaneously addresses multicollinearity, multiple comparisons, 
%      and clustered dependence. 
%
%      ROBUSTNESS TO OUTLIERS (HUBER TUNING):
%      While 'bootridge' assumes a Normal likelihood, it is numerically 
%      stabilized against outliers. The ridge penalty prevents individual 
%      coefficients from pivoting to accommodate high-leverage points [11]. 
%      Crucially, lambda is optimized with .632 bootstrap to minimize prediction
%      error based on mean Huber loss. The Huber loss function applies quadratic 
%      penalties to small residuals but only linear penalties to large ones, 
%      preventing outliers from distorting the tuning process. Because outliers 
%      are stochastically excluded from out-of-bag (OOB) samples in the .632 
%      bootstrap, any coefficient inflation caused by fitting local noise 
%      results in a high OOB error. The optimization therefore converges on 
%      a lambda value that reflects the stable covariance structure of the 
%      data rather than the influence of individual outliers [1, 2]. In 
%      multivariate settings, residual covariance is pooled across outcomes,
%      which further reduces the influence of outcome‑specific extreme values
%      on the inferred noise scale and Bayes factors.
%
%      REGULARIZATION AND MULTIPLE COMPARISONS: 
%      `bootridge` penalizes estimates via shrinkage rather than p-values. 
%      By pooling information across predictors to learn the global penalty, 
%      the model automatically adjusts skepticism to design complexity. 
%      "Partial Pooling" ensures Bayes factors are appropriately 
%      conservative without the significant reduction in statistical power 
%      often associated with classical post-hoc adjustments [6, 7].
%
%      UNCERTAINTY AND CLUSTERING:
%      The design effect (DEFF) is integrated at three levels:
%      1. Prior Learning: lambda is selected on the i.i.d. scale and 
%         divided by DEFF to "dilute" prior precision.
%      2. Scale Estimation: Residual variance is inflated by DEFF.
%      3. Inferential Shape: Inferential degrees of freedom are reduced 
%         (df_t = m/DEFF - trace(H)) to calibrate the Student-t tails 
%         for the effective sample size [12, 13].
%
%      DIAGNOSTIC ASSESSMENT:
%      Users should utilize `bootlm` for formal diagnostic plots (Normal 
%      Q-Q, Spread-Location, Cook’s Distance). These tools identify 
%      influential observations that may require inspection before or 
%      after ridge fitting.
%
%      See also: `bootstrp`, `boot`, `bootlm`, `bootbayes`, `bootwild`.
%
%  Bibliography:
%  [1] Delaney, N. J. & Chatterjee, S. (1986) Use of the Bootstrap and Cross-
%      Validation in Ridge Regression. Journal of Business & Economic Statistics,
%      4(2):255–262.
%  [2] Efron, B. & Tibshirani, R. J. (1993) An Introduction to the Bootstrap.
%      New York, NY: Chapman & Hall, pp. 247–252.
%  [3] Dickey, J. M. & Lientz, B. P. (1970) The Weighted Likelihood Ratio,
%      Sharp Hypotheses about Chances, the Order of a Markov Chain. Ann. Math.
%      Statist., 41(1):214–226. (Savage–Dickey)
%  [4] Morris, C. N. (1983) Parametric Empirical Bayes Inference: Theory and
%      Applications. Journal of the American Statistical Association, 78, 47–55.
%  [5] Wagenmakers, E.-J., et al. (2010) Bayesian Hypothesis Testing for
%      Psychologists. Psych. Science, 21(5):629–636. (Applied exposition)
%  [6] Gelman, A., Hill, J., & Yajima, M. (2012) Why we usually don't worry 
%      about multiple comparisons. J. Res. on Educ. Effectiveness, 5:189–211.
%  [7] Efron, B. (2010) Large-Scale Inference: Empirical Bayes Methods for 
%      Estimation, Testing, and Prediction. Cambridge University Press.
%  [8] Hastie, T., Tibshirani, R., & Friedman, J. (2009) The Elements of
%      Statistical Learning (2nd ed.). Springer. (Ridge, hat matrix, df)
%  [9] Ye, J. (1998) On Measuring and Correcting the Effects of Data Mining and
%      Model Selection. JASA, 93(441):120–131. (Generalized df)
% [10] Akaike, H. (1973) Information Theory and an Extension of the Maximum
%      Likelihood Principle. In: 2nd Int. Symp. on Information Theory. (AIC/KL)
% [11] Hoerl, A. E. & Kennard, R. W. (1970) Ridge Regression: Biased Estimation
%      for Nonorthogonal Problems. Technometrics, 12(1):55–67.
% [12] Satterthwaite, F. E. (1946) An Approximate Distribution of Estimates of
%      Variance Components. Biometrics Bulletin, 2(6):110–114.
% [13] Kenward, M. G. & Roger, J. H. (1997) Small Sample Inference for Fixed 
%      Effects from Restricted Maximum Likelihood. Biometrics, 53(3):983–997.
% [14] Vinod, H. D. (1987) Confidence Intervals for Ridge Regression Parameters.
%      In Time Series and Econometric Modelling, pp. 279–300.
% [15] Fisher, R. A. (1921) On the "Probable Error" of a Coefficient of
%      Correlation Deduced from a Small Sample. Metron, 1:3–32. (Fisher z)
% [16] Ly, A., Verhagen, J., & Wagenmakers, E.-J. (2016) Harold Jeffreys’s
%      Default Bayes Factor Hypothesis Tests: Explanation, Extension, and
%      Application in Psychology. J. Math. Psych., 72:19–32. (Correlation priors)
% [17] Neuhaus, J. M., & Segal, M. R. (1993) Design Effects for Binary 
%      Regression Models with Hierarchical Data. Biometrics, 49(3):971–979. 
%      (Generalized DEFF for regression).
% [18] Cameron, A. C., & Miller, D. L. (2015) A Practitioner's Guide to 
%      Cluster-Robust Inference. J. Hum. Resour., 50(2):317–372.
%
%  bootridge (version 2026.01.16)
%  Author: Andrew Charles Penn


function S = bootridge (Y, X, categor, nboot, alpha, L, deff, seed)

  % Check the number of input arguments provided
  if (nargin < 2)
    error (cat (2, 'bootridge: At least 2 input arguments, Y and X, required.'));
  end

  % Check that X and Y have the same number of rows
  if (size (Y, 1) ~= size (X, 1))
    error ('bootridge: the number of rows in X and Y must be the same');
  end

  % Omit any rows containing NaN or +/-Inf
  ridx = any ((isnan (cat (2, X, Y))) | (isinf (cat (2, X, Y))), 2);
  Y(ridx, :) = [];
  X(ridx, :) = [];

  % Get dimensions of the data
  [m, n] = size(X);
  q = size (Y, 2);

  % Check that the first column is X are all equal to 1, if not create one
  if ( ~all (X(:, 1) == 1) )
    X = cat (2, ones (m, 1), X);
    n = n + 1;
  end

  % If categor is not provided, set it to empty
  if ( (nargin < 3) || isempty (categor) )
    categor = [];
  else
    if ( any (strcmpi(categor, {'all', '*'})) )
      categor = (2:n);
    end
    if ( (~ isnumeric (categor)) || (sum (size (categor) > 1) > 1) || ...
         (any (isnan (categor))) || (any (isinf (categor))) || ...
         (any (categor < 1)) || (~ all (categor == abs (fix (categor)))) )
      error (cat (2, 'bootridge: categor should be a vector of column', ...
                     ' numbers corresponding to categorical variables'));
    end
    if ( ~ all (categor > 1) )
      error ('bootridge: The intercept should not be included in categor.')
    end
    if ( any (categor > n) )
      error ('bootridge: Numbers in categor exceed the number of columns in X');
    end
  end

  % If nboot is not specified, set it to 100.
  if ( (nargin < 4) || isempty (nboot) )
    nboot = 100;
  else
    if ((nboot <= 0) || (nboot ~= fix (nboot)) || isinf (nboot) || ...
         isnan (nboot) || (numel (nboot) > 1))
      error ('bootridge: nboot must be a finite positive integer');
    end
  end

  % If alpha is not specified, set it to .05.
  if ( (nargin < 5)|| isempty (alpha) )
    alpha = .05;
  else
    if ( any (alpha <= 0) || any (alpha >= 1) || (numel (alpha) > 1) )
      error ('bootridge: Value of alpha must be between 0 and 1');
    end
  end

  % If a hypothesis matrix (L) is not provided, set it to empty
  if ( (nargin < 6) || isempty (L) )
    L = [];
    c = 0;
  else
    if (~ isempty (L))
      if (size (L, 1) ~= n)
        error (cat (2, 'bootridge: If not empty, L must have the same', ...
                       ' number of rows as columns in X.'));
      end
      c = size (L, 2);
    else
      c = 0;
    end
  end

  % If DEFF not specified, set it to 1
  if ( (nargin < 7) || isempty (deff) )
    deff = 1;
  else
    if ( (deff <= 0) || isinf (deff) || isnan (deff) || (numel (deff) > 1) )
      error ('bootridge: DEFF must be a finite scalar value > 0');
    end
  end

  % If seed not specified, set it to 1
  if ( (nargin < 8) || isempty (seed) )
    seed = 1;
  else
    if ( isinf (seed) || isnan (seed) || (numel (seed) > 1) || ...
         seed ~= fix(seed))
      error ('bootridge: The seed must be a finite integer');
    end
  end

  % Check the number of output arguments requested
  if (nargout > 1)
    error ('bootridge: Only 1 output argument can be requested.');
  end

  % Create the penalty matrix - ridge regression will shrink all but intercept.
  % The penalty weight of each predictor term equals its variance. This makes
  % ridge shrinkage equivalent to what you would get if predictors were
  % standardized, ensuring the ridge parameter (lambda) applies uniformly across
  % predictors regardless of scale.
  P = cat (2, 0, var (X(:,2:end), 0, 1));

  % Evaluate categor input argument.
  if (~ isempty (categor))   
    % Set P(k) to 1 where the predictor term corresponds to a categorical
    % variable. Categorical variable coding is exempt from penalty scaling.
    P(categor) = 1;
  end

  % Convert the vector of penalties to a diaganol matrix
  P = diag (P);

  % Approximate lambda0 is the variance ratio of the residuals and coefficients.
  % This estimator scale invariant since it uses the standardized data. The
  % initial lambda0 estimate serves only as a scale‑invariant anchor for the
  % bootstrap‑optimized ridge parameter and does not determine the final amount
  % of shrinkage.
  z_score = @(A) bsxfun (@rdivide, bsxfun (@minus, A, mean (A)), std (A, 0));
  YS = z_score (Y);
  mask = true (1, n); mask([1, categor]) = false;
  XS = X; XS(:, mask) = z_score (XS(:, mask));
  bOLS = XS \ YS;
  if ( (n > 2) &&  (all (var (bOLS(2:end,:), 1) > eps)) )
    % Inverted SNR ratio estimator
    lambda0 = sum (sum ((YS - XS * bOLS).^2) ./ ...
              ( max (m - rank (XS), 1)  * var (bOLS(2:end,:), 1)), 2) / q;
  else
    % Hoerl–Kennard estimator
    lambda0 = sum ((sum ((YS - XS * bOLS).^2) * (n - 1)) ./ ...
              ( max (m - rank (XS), 1) * sum (bOLS(2:end, :).^2, 1)), 2) / q;
  end
  if (lambda0 < eps)
    lambda0 = NaN;
  end

  % Use the median absolute deviation of the residuals to set the sensitivity
  % threshold subsequently used for the Huber loss function. The cutoff k is
  % in residual units; default k = 1.345 corresponds to ~95% efficiency under
  % errors from a standard normal distribution.
  RS = YS - XS * bOLS;
  ks = max (1.345 * median (abs (RS (:) - median (RS(:)))) / 0.6745, 1e-6);

  % Objective for lambda using .632 bootstrap prediction error
  % Standardizing outcomes (YS) ensures equal weight across multivariate 
  % dimensions, while Huber loss (k) provides outlier resistance.
  obj_func = @(lambda) booterr632 (YS, X, lambda, P, ks, nboot, seed);

  % Golden-section search for optimal lambda by .632 bootstrap prediction error
  smax = svds (X, 1);                    % returns the largest singular value
  amin = log10 (smax^2 * eps);           % minimum a for well-conditioned system
  bmax = log10 (smax^2);                 % maximum b for well-conditioned system
  tol = 0.1;
  expand = 1;                            % one order of magnitude usually enough
  max_expand = 10;
  while (expand < max_expand )
    if (isfinite (lambda0))
      a = max (amin, log10 (lambda0) - expand);   % set the lower bound
      b = min (bmax, log10 (lambda0) + expand);   % set the upper bound
    else
      a = amin; b = bmax;
    end
    [lambda, iter] = gss (obj_func, a, b, tol);
    if ( all (abs (log10 (lambda) - [a, b]) > tol ) )
      break
    else
      expand = expand + 1;
    end
  end

  % Heuristic correction to lambda (prior precision) for the design effect.
  % Empirical-Bayes ridge learns lambda as an inverted estimator-scale SNR:
  %    lambda_iid ≈ sigma^2 / tau2_iid
  % where sigma^2 (a.k.a. Sigma_Y_hat) is residual variance (data space)
  % and tau^2 (a.k.a. tau2_hat) is the prior variance (parameter space).
  %
  % Under clustering, the information density is reduced. To maintain 
  % a consistent prior scale, the prior variance must be inflated:
  %    tau2_true = DEFF * tau2_iid
  %
  % Hence the EB precision learned under an i.i.d. assumption is too large:
  %    lambda_true = sigma^2 / tau2_true = lambda_iid / DEFF
  %
  % Thus, our apparent prior precision (lambda) under i.i.d. must be scaled
  % down by a factor of DEFF to prevent over-regularization.
  lambda = lambda / deff;

  % Regression coefficient and the effective degrees of freedom for ridge
  % regression penalized using the optimized (and corrected) lambda
  A = X' * X + lambda * P;
  Beta = A \ (X' * Y);                          % n x q coefficient matrix
  df_lambda = m  - trace (A \ (X' * X));        % equivalent to m - trace (H)
  if (df_lambda < 1)
    error ('bootridge: df_lambda < 1; check the model is correctly specified')
  end
  df_lambda = max (df_lambda, 1);

  % Calculate the residuals for the ridge fit using the optimal lambda
  resid = Y - X * Beta;                               % resid is m x q

  % Huber-weighted residuals based on equivalent k for each outcome on their
  % original scale.
  %k = max (1.345 * median (abs (bsxfun (@minus, resid, ...
  %                                      median (resid, 1))), 1) / 0.6745, 1e-6);
  %[mean_loss, weights] = mean_huber_loss (resid, k);  % weights is m x q
  %resid_w = resid .* weights;
  
  % Residual (co)variance (q x q) scaled by the design effect (Deff) to
  % propagate to all subsequent calculations of the prior and posterior.
  Sigma_Y_hat = deff * (resid' * resid) / df_lambda;

  % Prior (co)variance (matrix for q > 1)
  % The prior scale is effectively inflated by a factor of Deff^2
  tau2_hat = Sigma_Y_hat / lambda;

  % Posterior covariance (diagonal block) for the coefficients within outcome j;
  % Sigma_Beta{j} = Sigma_Y_hat(j,j) * invA
  invA = A \ eye(n);
  Sigma_Beta = arrayfun (@(j) Sigma_Y_hat(j, j) * invA, (1 : q), ...
                        'UniformOutput', false);

  % Distribution functions.
  % Student's (t) distribution:
  % A t-distribution for the prior and posterior is a mathematical approximation
  % in this empirical Bayes framework to having placed an Inverse-Gamma prior on
  % the variance and integrating it out.
  if ((exist ('betaincinv', 'builtin')) || (exist ('betaincinv', 'file')))
    distinv = @(p, df) sign (p - 0.5) * ...
                sqrt ( df ./ betaincinv (2 * min (p, 1 - p), df / 2, 0.5) - df);
  else
    % Earlier versions of Matlab do not have betaincinv
    % Instead, use betainv from the Statistics and Machine Learning Toolbox
    try 
      distinv = @(p, df) sign (p - 0.5) * ...
                  sqrt ( df ./ betainv (2 * min (p, 1 - p), df / 2, 0.5) - df);
    catch
      % Use critical values from the Normal distribution if either betaincinv
      % or betainv are not available
      distinv = @(p, df) sqrt (2) * erfinv (2 * p - 1);
      warning ('bootridge:', ...
      'Could not create studinv function; intervals will use z critical value');
    end
  end
  distpdf = @(t, mu, v, df) (exp (gammaln ((df + 1) / 2) - ...
                             gammaln (df / 2)) ./ sqrt(df * pi * v)) .* ...
                             (1 + (t - mu).^2 ./ (df .* v)).^(-(df + 1) / 2);
  % Normal (z) distribution:
  %distinv = @(p, df) sqrt (2) * erfinv (2 * p - 1);
  %distpdf = @(z, mu, v, df) exp (-0.5 * ((z - mu).^2) ./ v) ./ sqrt (2 * pi * v);
  
  % Set critical value for credibility intervals
  %critval = stdnorminv (1 - alpha / 2);   % Use Normal z distribution
  df_t = m / deff - trace (A \ (X' * X));  % Use Student's t distribution
  critval = distinv (1 - alpha / 2, df_t);

  % Calculation of credibility intervals
  if (c < 1)

    % Calculation of credibility intervals for model coefficients
    CI_lower = zeros (n, q);
    CI_upper = zeros (n, q);
    for j = 1:q
      se_j = sqrt (diag (Sigma_Beta{j}));
      CI_lower(:,j) = Beta(:,j) - critval .* se_j;
      CI_upper(:,j) = Beta(:,j) + critval .* se_j;
    end

    % Calculations for reporting Bayes factors for regression coefficients.
    % Prior variance and standard deviation for each coefficient (rows) and
    % outcome (columns). Report mean and standard deviation of the normal 
    % distribution used for the prior.
    ridx = false (n, 1); ridx(1) = true;
    V0 = bsxfun (@rdivide, diag (tau2_hat)', diag (P)); V0(ridx,:) = Inf; 
    prior = arrayfun (@(v) sprintf('t (0, %#.3g, %#.3g)', ...
                      sqrt (v), df_t), V0, 'UniformOutput', false);
    %prior = arrayfun (@(v) sprintf('N (0, %#.3g)', ...
    %                  sqrt (v)), V0, 'UniformOutput', false);
    prior(ridx,:) = {'U (-Inf, Inf)'};

    % Marginal posterior variances for each coefficient-outcome pair
    V1 = diag (invA) * diag (Sigma_Y_hat)';

    % Marginal posterior density at 0 for each coefficient/outcome
    % Note that the third input argumemt is variance, not standard deviation.
    pH1 = distpdf (0, Beta, V1, df_t);

  else

    % Calculation of credibility intervals for model-based estimates
    mu = L' * Beta;  % c x q matrix
    CI_lower = zeros (c, q);
    CI_upper = zeros (c, q);
    for j = 1:q
      % Posterior variance for linear estimates for outcome j
      se_j = sqrt (diag (L' * Sigma_Beta{j} * L));
      CI_lower(:,j) = mu(:,j) - critval .* se_j;
      CI_upper(:,j) = mu(:,j) + critval .* se_j;
    end

    % Calculations for reporting Bayes factors for linear estimates.
    % Prior variance and standard deviation for each estimate (rows) and
    % outcome (columns). Report mean and standard deviation of the normal 
    % distribution used for the prior.
    ridx = ( abs (L(1,:)') > eps );
    P_L = L' * pinv (P) * L;
    V0 = bsxfun (@times, diag (tau2_hat)', diag (P_L)); V0(ridx, :) = Inf;
    prior = arrayfun (@(v) sprintf('t (0, %#.3g, %#.3g)', ...
                      sqrt (v), df_t), V0, 'UniformOutput', false);
    %prior = arrayfun (@(v) sprintf('N (0, %#.3g)', ...
    %                  sqrt (v)), V0, 'UniformOutput', false);
    prior(ridx, :) = repmat ({'U (-Inf, Inf)'}, nnz (ridx), q);

    % Marginal posterior variances for each linear estimate/outcome
    invA_L = L' * invA * L;
    V1 = diag (invA_L) * diag (Sigma_Y_hat)';              % c x q

    % Marginal posterior density at 0 for each linear estimate/outcome
    % Note that the third input argumemt is variance, not standard deviation.
    pH1 = distpdf (0, mu, V1, df_t);

  end

  % Marginal prior density at 0 for each estimate or coefficient per outcome
  %pH0 = (2 * pi * V0).^(-0.5);      % Applies only to a Normal distribution
  pH0 = distpdf (0, 0, V0, df_t);

  % Bayes factor (Savage–Dickey ratio): relative plausibility (density) of the
  % coefficient or estimate being exactly 0 under the prior compared to under
  % the posterior.
  % BF10 > 1: The data have made observing Beta = 0 LESS plausible.
  %           This is evidence in favour of the alternative hypothesis.
  % BF10 < 1: The data have made observing Beta = 0 MORE plausible.
  %           This is evidence in favour of the null hypothesis.
  BF10   = bsxfun (@rdivide, pH0, pH1); BF10(ridx,:) = NaN;
  lnBF10 = log (BF10);                 % ln(1) = 0, ln(0.3) ~= -1, ln(3) ~= +1

  % Posterior correlation between outcomes
  d = sqrt (diag (Sigma_Y_hat));
  R = Sigma_Y_hat ./ (d * d');
  R = R(tril (true (size (R)), -1));

  % Credible intervals for correlations between outcomes (closed-form)
  if (q > 1)
 
    % Fisher's z-transform with numerical guard
    Z = atanh (min (max (R, -1 + eps), 1 - eps));

    % Standard error of Z
    SE_z = 1 / sqrt (max (df_t, 4) - 3);

    % Symmetric credible intervals for Z, then back-transform to R
    R_CI_lower = tanh (Z - critval * SE_z);
    R_CI_upper = tanh (Z + critval * SE_z);

    % Prior: Uniform(-1,1) on R  ==>  Logistic(0, 1/2) on Fisher-z
    % A logistic distribution on Z is induced by a flat prior on the correlation
    % coefficient, yielding fully closed‑form, non‑arbitrary Bayes factors for
    % residual correlations. Density at zero under the prior:
    pH0_Z = 0.5;

    % Posterior density at zero under t-marginal on Fisher-z
    pH1_Z = distpdf (0, Z, SE_z.^2, df_t);

    % Bayes factors (Savage–Dickey ratio) for the correlation coefficient
    BF10_R   = bsxfun (@rdivide, pH0_Z, pH1_Z);
    lnBF10_R = log (BF10_R);

    % Create labels
    [I, J] = find (tril (true (q), -1));
    labels = arrayfun (@(i,j) sprintf ('R_%d,%d', j, i), ...
                       I, J, 'UniformOutput', false);

    % Assemble table-like cell array for correlations
    R_table = cat (1, ...
     {'Correlation', 'R', 'CI_lower', 'CI_upper', 'BF10', 'lnBF10'}, ...
     [labels(:), num2cell([R(:), R_CI_lower(:), R_CI_upper(:), ...
                               BF10_R(:), lnBF10_R(:)])])';

  end

  % Pack results
  if (c < 1); S.Coefficient = Beta; else; S.Estimate = mu; end;
  S.CI_lower = CI_lower;
  S.CI_upper = CI_upper;
  S.BF10 = BF10;
  S.lnBF10 = lnBF10;
  S.prior = prior;
  S.lambda = lambda;
  S.Sigma_Y_hat = Sigma_Y_hat;
  S.df_lambda = df_lambda;
  S.tau2_hat = tau2_hat;
  S.Sigma_Beta = Sigma_Beta;
  S.nboot = nboot;
  S.Deff = deff;
  S.tol = tol;
  S.expand = expand;
  S.iter = iter;
  if (q > 1); S.R_table = R_table; end

  % Display summary
  if (nargout == 0)
    fprintf (cat (2, '\n Empirical Bayes Ridge Regression (.632 Bootstrap',...
                     ' Tuning) - Summary\n ******************************', ...
                     '*************************************************\n'));
    fprintf ('\n Bootstrap optimized ridge tuning constant (lambda): %.6g\n', ...
             lambda);
    fprintf ('\n Effective residual degrees of freedom: %.3f\n', df_lambda);
    fprintf ('\n Number of outcomes (q): %d\n', q);
    fprintf ('\n Design effect (Deff): %.2g\n', deff);
    if (q > 1)
      fprintf ('\n Residual covariance (sigma^2):\n'); disp (Sigma_Y_hat);
    else
      fprintf ('\n Residual variance (sigma^2): %.3f\n', Sigma_Y_hat);
    end

    % Correlations between outcomes
    if (q > 1)
      fprintf (cat (2, '\n %.3g%% credible intervals and Bayes factors', ...
                       ' for correlations between outcomes:\n'), ...
                       100* (1 - alpha));
      fprintf (cat (2, '\n Correlation   CI_lower      CI_upper      ', ...
                         'lnBF10        Outcomes\n'));
      for i = 1:q*(q-1)*0.5
        fprintf (' %#-+10.4g    %#-+10.4g    %#-+10.4g    %#-+10.4g    %s\n', ...
                 R(i), R_CI_lower(i), R_CI_upper(i), lnBF10_R(i), labels{i});
      end
    end

    % Coefficients and linear estimates
    if (c < 1)
      fprintf (cat (2, '\n %.3g%% credible intervals and Bayes factors', ...
                       ' for regression coefficients:\n'), ...
                       100* (1 - alpha));
      for j = 1:q
        fprintf (cat (2, '\n Outcome %d:\n Coefficient   CI_lower      ', ...
                         'CI_upper      lnBF10        Prior\n'), j);
        for k = 1:n
          fprintf (' %#-+10.4g    %#-+10.4g    %#-+10.4g    %#-+10.4g    %s\n', ...
                  Beta(k, j), CI_lower(k, j), CI_upper(k, j), lnBF10(k, j), ...
                  prior{k, j});
        end
      end
    else
      fprintf (cat (2, '\n %.3g%% credible intervals and Bayes factors', ...
                       ' for linear estimates:\n'), ...
                       100* (1 - alpha));
      for j = 1:q
        fprintf (cat (2, '\n Outcome %d:\n Estimate      CI_lower      ', ...
                         'CI_upper      lnBF10        Prior\n'), j);
        for k = 1:c
          fprintf (' %#-+10.4g    %#-+10.4g    %#-+10.4g    %#-+10.4g    %s\n', ...
                  mu(k, j), CI_lower(k, j), CI_upper(k, j), lnBF10(k, j), ...
                  prior{k, j});
        end
      end
    end
    fprintf('\n');

  end

end

%--------------------------------------------------------------------------

%% FUNCTION FOR .632 BOOTSTRAP ESTIMATOR OF PREDICTION ERROR

function PRED_ERR = booterr632 (Y, X, lambda, P, k, nboot, seed)

  % Efron and Tibshirani (1993) An Introduction to the Bootstrap. New York, NY:
  %  Chapman & Hall. pg 247-252

  % Anonymous function for ridge regression (returns n x q)
  ridge = @(Y, X) (X' * X + lambda * P) \ (X' * Y);

  % The following bootstrap approach uses bootknife resampling to avoid empty
  % OOB samples. Resampling is also balanced to reduce Monte Carlo error.
  [BOOTSTAT, jnk, jnk, BOOTOOB] = bootstrp (nboot, ridge, Y, X, 'match', ...
                                            true, 'loo', true, 'seed', seed);

  % Calculate the number of outcomes (q > 1 for multivariate)
  q = size (Y, 2);

  % Simple bootstrap estimate of error (S_ERR)
  % (S_ERR is equivalent to MSEP in Delaney and Chatterjee, 1986)
  S_ERR = mean (arrayfun (@(b) ...
          mean_huber_loss (Y(BOOTOOB{b}, :) - X(BOOTOOB{b}, :) * ...
                           reshape (BOOTSTAT(b, :), [], q), k), 1:nboot));

  % Apparent error in resamples (A_ERR)
  A_ERR = mean_huber_loss (Y - X * ridge (Y, X), k);

  % Optimism in apparent error
  OPTIM = .632 * (S_ERR - A_ERR);

  % The bootstrap .632 estimator of prediction error
  PRED_ERR = A_ERR + OPTIM;

end

%--------------------------------------------------------------------------

%% FUNCTION FOR MEAN HUBER LOSS (ROBUST PREDICTION ERROR CRITERION)

function [mean_loss, weights] = mean_huber_loss (resid, k)

  % If k is not provided, set scalar k for the standardized scale (SD = 1)
  % k must be size 1 x q
  if (nargin < 2); k = 1.345; end;

  % is_small will be m x q, comparing each resid to its specific k
  abs_resid = abs (resid);
  is_small = abs_resid <= k;

  % Calculate loss matrix
  loss = is_small .* (0.5 * resid.^2) + ...
        ~is_small .* bsxfun (@minus, bsxfun (@times, k, abs_resid), 0.5 * k.^2);

  % Total error is the mean of all elements
  mean_loss = mean (loss(:));

  % Return Huber weights (m x q)
  if (nargout > 1)
    weights = is_small + ~is_small .* bsxfun (@rdivide, k, abs_resid + eps);
  end

end

%--------------------------------------------------------------------------

% FUNCTION FOR GOLDEN-SECTION SEARCH MINIMIZATION OF AN OBJECTIVE FUNCTION

function [lambda, iter] = gss (f, a, b, tol)

  % Algorithm based on https://en.wikipedia.org/wiki/Golden-section_search
  invphi = (sqrt (5) - 1) / 2;
  iter = 0;
  while ((b - a) > tol )
    c = b - (b - a) * invphi;
    d = a + (b - a) * invphi;
    if f(10.^c) < f(10.^d)
      b = d;
    else
      a = c;
    end
    iter = iter + 1;
  end
  lambda = 10.^((b + a)/2);

end

%--------------------------------------------------------------------------

%!demo
%!
%! % Simple linear regression. The data represents salaries of employees and
%! % their years of experience, modified from Allena Venkata. The salaries are
%! % in units of 1000 dollars per annum.
%!
%! years = [1.20 1.40 1.60 2.10 2.30 3.00 3.10 3.30 3.30 3.80 4.00 4.10 ...
%!               4.10 4.20 4.60 5.00 5.20 5.40 6.00 6.10 6.90 7.20 8.00 8.30 ...
%!               8.80 9.10 9.60 9.70 10.40 10.60]';
%! salary = [39 46 38 44 40 57 60 54 64 57 63 56 57 57 61 68 66 83 81 94 92 ...
%!           98 101 114 109 106 117 113 122 122]';
%!
%! bootridge (salary, years);
%! 
%! % We can see from the intercept that the starting starting salary is $25.2 K
%! % and that salary increase per year of experience is $9.4 K.

%!demo
%!
%! % Two-sample unpaired test on independent samples.
%!
%! score = [54 23 45 54 45 43 34 65 77 46 65]';
%! gender = {'male' 'male' 'male' 'male' 'male' 'female' 'female' 'female' ...
%!           'female' 'female' 'female'}';
%!
%! % Difference between means
%! % Note that the 'dim' argument in `bootlm` automatically changes the default
%! % coding to simple contrasts, which are centered.
%! [STATS, BOOTSTAT, AOVSTAT, PRED_ERR, MAT]  = bootlm (score, gender, ...
%!  'display', 'on', 'varnames', 'gender', 'dim', 1, 'posthoc', 'trt_vs_ctrl');
%! bootridge (MAT.Y, MAT.X, 2);
%!
%! % Group means
%! [STATS, BOOTSTAT, AOVSTAT, PRED_ERR, MAT]  = bootlm (score, gender, ...
%!                            'display', 'off', 'varnames', 'gender', 'dim', 1);
%! bootridge (MAT.Y, MAT.X, 2, [], [], MAT.L);

%!demo
%!
%! % One-way repeated measures design.
%! % The data is from a study on the number of words recalled by 10 subjects
%! % for three time condtions, in Loftus & Masson (1994) Psychon Bull Rev. 
%! % 1(4):476-490, Table 2.
%!
%! words = [10 13 13; 6 8 8; 11 14 14; 22 23 25; 16 18 20; ...
%!          15 17 17; 1 1 4; 12 15 17;  9 12 12;  8 9 12];
%! seconds = [1 2 5; 1 2 5; 1 2 5; 1 2 5; 1 2 5; ...
%!            1 2 5; 1 2 5; 1 2 5; 1 2 5; 1 2 5;];
%! subject = [ 1  1  1;  2  2  2;  3  3  3;  4  4  4;  5  5  5; ...
%!             6  6  6;  7  7  7;  8  8  8;  9  9  9; 10 10 10];
%!
%! % Frequentist framework: wild bootstrap of linear model, with orthogonal
%! % polynomial contrast coding followed up with treatment vs control
%! % hypothesis testing.
%! [STATS, BOOTSTAT, AOVSTAT, PRED_ERR, MAT] = bootlm (words, ...
%!                  {subject,seconds},  'display', 'on', 'varnames', ...
%!                  {'subject','seconds'}, 'model', 'linear', 'contrasts', ...
%!                  'poly', 'dim', 2, 'posthoc', 'trt_vs_ctrl');
%!
%! % Ridge regression and bayesian analysis of posthoc comparisons
%! bootridge (MAT.Y, MAT.X, '*', 200, 0.05, MAT.L);
%!
%! % Frequentist framework: wild bootstrap of linear model, with orthogonal
%! % polynomial contrast coding followed up estimating marginal means.
%! [STATS, BOOTSTAT, AOVSTAT, PRED_ERR, MAT] = bootlm (words, ...
%!                  {subject,seconds},  'display', 'on', 'varnames', ...
%!                  {'subject','seconds'}, 'model', 'linear', 'contrasts', ...
%!                  'poly', 'dim', 2);
%!
%! % Ridge regression and bayesian analysis of model estimates. Note that group-
%! % mean Bayes Factors are NaN under the flat prior on the intercept whereas
%! % the contrasts we just calculated had proper Normal priors.
%! bootridge (MAT.Y, MAT.X, '*', 200, 0.05, MAT.L);

%!demo
%!
%! % One-way design with continuous covariate. The data is from a study of the
%! % additive effects of species and temperature on chirpy pulses of crickets,
%! % from Stitch, The Worst Stats Text eveR
%!
%! pulse = [67.9 65.1 77.3 78.7 79.4 80.4 85.8 86.6 87.5 89.1 ...
%!          98.6 100.8 99.3 101.7 44.3 47.2 47.6 49.6 50.3 51.8 ...
%!          60 58.5 58.9 60.7 69.8 70.9 76.2 76.1 77 77.7 84.7]';
%! temp = [20.8 20.8 24 24 24 24 26.2 26.2 26.2 26.2 28.4 ...
%!         29 30.4 30.4 17.2 18.3 18.3 18.3 18.9 18.9 20.4 ...
%!         21 21 22.1 23.5 24.2 25.9 26.5 26.5 26.5 28.6]';
%! species = {'ex' 'ex' 'ex' 'ex' 'ex' 'ex' 'ex' 'ex' 'ex' 'ex' 'ex' ...
%!            'ex' 'ex' 'ex' 'niv' 'niv' 'niv' 'niv' 'niv' 'niv' 'niv' ...
%!            'niv' 'niv' 'niv' 'niv' 'niv' 'niv' 'niv' 'niv' 'niv' 'niv'};
%!
%! % Estimate regression coefficients using 'anova' contrast coding 
%! [STATS, BOOTSTAT, AOVSTAT, PRED_ERR, MAT]  = bootlm (pulse, ...
%!                           {temp, species}, 'model', 'linear', ...
%!                           'continuous', 1, 'display', 'on', ...
%!                           'varnames', {'temp', 'species'}, ...
%!                           'contrasts', 'anova');
%!
%! % Ridge regression and bayesian analysis of regression coefficients
%! % MAT.X: column 1 is intercept, column 2 is temp (continuous), column 3 
%! % is species (categorical).
%! bootridge (MAT.Y, MAT.X, 3, 200, 0.05);

%!demo
%!
%! % Variations in design for two-way ANOVA (2x2) with interaction. 
%!
%! % Arousal was measured in rodents assigned to four experimental groups in a
%! % between-subjects design with two factors: group (lesion/control) and
%! % stimulus (fearful/neutral). In this design, each rodent is allocated to one 
%! % combination of levels in group and stimulus, and a single measurment of
%! % arousal is made. The question we are asking here is, does the effect of a
%! % fear-inducing stimulus on arousal depend on whether or not rodents had a
%! % lesion?
%!
%! group = {'control' 'control' 'control' 'control' 'control' 'control' ...
%!          'lesion'  'lesion'  'lesion'  'lesion'  'lesion'  'lesion'  ...
%!          'control' 'control' 'control' 'control' 'control' 'control' ...
%!          'lesion'  'lesion'  'lesion'  'lesion'  'lesion'  'lesion'};
%! 
%! stimulus = {'fearful' 'fearful' 'fearful' 'fearful' 'fearful' 'fearful' ...
%!             'fearful' 'fearful' 'fearful' 'fearful' 'fearful' 'fearful' ...
%!             'neutral' 'neutral' 'neutral' 'neutral' 'neutral' 'neutral' ...
%!             'neutral' 'neutral' 'neutral' 'neutral' 'neutral' 'neutral'};
%!
%! arousal = [0.78 0.86 0.65 0.83 0.78 0.81 0.65 0.69 0.61 0.65 0.59 0.64 ...
%!            0.54 0.6 0.67 0.63 0.56 0.55 0.645 0.565 0.625 0.485 0.655 0.515];
%!
%! % Fit between-subjects design
%! [STATS, BOOTSTAT, AOVSTAT, PRED_ERR, MAT] = bootlm (arousal, ...
%!                                   {group, stimulus}, 'seed', 1, ...
%!                                   'display', 'on', 'contrasts', 'simple', ...
%!                                   'model', 'full', ...
%!                                   'method', 'bayes', ...
%!                                   'varnames', {'group', 'stimulus'});
%!
%! % Ridge regression and bayesian analysis of regression coefficients
%! % MAT.X: column 1 is intercept, column 2 is temp (continuous), column 3 
%! % is species (categorical).
%! bootridge (MAT.Y, MAT.X, '*', 200, 0.05);
%!
%! % Now imagine the design is repeated stimulus measurements in each rodent
%! ID = [1 2 3 4 5 6 7 8 9 10 11 12 1 2 3 4 5 6 7 8 9 10 11 12]';
%!
%! % Fit model including ID as a blocking-factor
%! [STATS, BOOTSTAT, AOVSTAT, PRED_ERR, MAT] = bootlm (arousal, ...
%!                                   {ID, group, stimulus}, 'seed', 1, ...
%!                                   'display', 'on', 'contrasts', 'simple', ...
%!                                   'model', [1 0 0; 0 1 0; 0 0 1; 0 1 1], ...
%!                                   'method', 'bayes', ...
%!                                   'varnames', {'ID', 'group', 'stimulus'});
%!
%! % Ridge regression and bayesian analysis of regression coefficients
%! % MAT.X: column 1 is intercept, column 2 is temp (continuous), column 3 
%! % is species (categorical).
%! bootridge (MAT.Y, MAT.X, '*', 200, 0.05);

%!demo
%!
%! % Analysis of nested one-way design.
%!
%! % Nested model example from:
%! % https://www.southampton.ac.uk/~cpd/anovas/datasets/#Chapter2
%!
%! data = [4.5924 7.3809 21.322; -0.5488 9.2085 25.0426; ...
%!         6.1605 13.1147 22.66; 2.3374 15.2654 24.1283; ...
%!         5.1873 12.4188 16.5927; 3.3579 14.3951 10.2129; ...
%!         6.3092 8.5986 9.8934; 3.2831 3.4945 10.0203];
%!
%! clustid = [1 3 5; 1 3 5; 1 3 5; 1 3 5; ...
%!            2 4 6; 2 4 6; 2 4 6; 2 4 6];
%!
%! group = {'A' 'B' 'C'; 'A' 'B' 'C'; 'A' 'B' 'C'; 'A' 'B' 'C'; ...
%!          'A' 'B' 'C'; 'A' 'B' 'C'; 'A' 'B' 'C'; 'A' 'B' 'C'};
%!
%! [STATS, BOOTSTAT, AOVSTAT, PREDERR, MAT] = bootlm (data, {group}, ...
%!      'clustid', clustid, 'seed', 1, 'display', 'off', 'contrasts', ...
%!      'helmert', 'method', 'bayes', 'dim', 1, 'posthoc', 'trt_vs_ctrl');
%!
%! % Fit a cluster-robust empirical Bayes model
%! g = mean (accumarray (clustid(:), 1, [], @sum));    % g is mean cluster size
%! bootridge (MAT.Y, MAT.X, '*', 200, 0.05, MAT.L, g); % Upperbound DEFF is m

%!demo
%!
%! % Generic univariate example with auto-added intercept
%! m = 40;
%! x = linspace (-1, 1, m).';
%! X = x;                          % No intercept column; function will add it
%! beta = [2; 0.5];
%! randn ('twister', 123);
%! y = beta(1) + beta(2) * x + 0.2 * randn (m, 1);
%! % Run bootridge with a small bootstrap count for speed
%! bootridge (y, X, [], 100, 0.05, [], 1, 123);
%! % Please be patient, the calculations will be completed soon...

%!demo
%!
%! % Generic multivariate outcome example with explicit intercept
%! m = 35;
%! x = linspace (-2, 2, m).';
%! X = [ones(m,1), x];
%! B = [1.5,  2.0;    % intercepts for 2 outcomes
%!      0.6, -0.3];   % slopes for 2 outcomes
%! randn ('twister', 123);
%! E = 0.25 * randn (m, 2);
%! Y = X * B + E;
%! % Run bootridge with small bootstrap count
%! bootridge (Y, X, [], 100, 0.10, [], 1, 321);
%! % Please be patient, the calculations will be completed soon...

%!test
%! % Basic functionality: univariate, intercept auto-add, field shapes
%! m = 30;
%! x = linspace (-1, 1, m).';
%! X = x;                           % No intercept provided
%! randn ('twister', 123);
%! y = 1.0 + 0.8 * x + 0.1 * randn (m,1);
%! S = bootridge (y, X, [], 200, 0.05, [], 1.1, 777);
%! % Check expected fields and sizes
%! assert (isfield (S, 'Coefficient'));
%! assert (~ isfield (S, 'Estimate'));
%! assert (size (S.Coefficient, 2) == 1);
%! assert (size (S.Coefficient, 1) == 2);     % intercept + slope
%! assert (isfinite (S.lambda) && (S.lambda > 0));
%! assert (isfinite (S.df_lambda) && (S.df_lambda > 0) && ...
%!         (S.df_lambda <= m));
%! assert (all (S.CI_lower(:) <= S.Coefficient(:) + eps));
%! assert (all (S.CI_upper(:) + eps >= S.Coefficient(:)));
%! assert (isfinite (S.Sigma_Y_hat) && (S.Sigma_Y_hat > 0));
%! assert (iscell (S.Sigma_Beta) && (numel (S.Sigma_Beta) == 1));
%! assert (all (size (S.Sigma_Beta{1}) == [2, 2]));
%! assert (S.nboot == 200);
%! assert (S.Deff == 1.1);
%! assert (S.expand >= 1);

%!test
%! % Hypothesis matrix L: return linear estimate instead of coefficients
%! m = 28;
%! x = linspace (-1.5, 1.5, m).';
%! X = [ones(m,1), x];              % Explicit intercept is first column
%! randn ('twister', 123);
%! y = 3.0 + 0.4 * x + 0.15 * randn (m,1);
%! % Contrast to extract only the slope (second coefficient)
%! L = [0; 1];
%! S = bootridge (y, X, [], 100, 0.10, L, 1, 99);
%! assert (~ isfield (S, 'Coefficient'));
%! assert (isfield (S, 'Estimate'));
%! assert (all (size (S.Estimate) == [1, 1]));
%! assert (all (size (S.CI_lower) == [1, 1]));
%! assert (all (size (S.CI_upper) == [1, 1]));
%! assert (all (size (S.BF10 ) == [1, 1]));
%! assert (iscell (S.prior) && all (size (S.prior) == [1, 1]));
%! assert (S.expand >= 1);

%!test
%! % Categorical predictor supplied via CATEGOR (no scaling)
%! m = 36;
%! % Two-level factor coded as centered +/-0.5 (column 2), plus a continuous
%! g = repmat ([ -0.5; 0.5 ], 18, 1);
%! x = linspace (-2, 2, m).';
%! X = [ones(m,1), g, x];
%! beta = [1.0; 0.7; -0.2];
%! randn ('twister', 123);
%! y = X * beta + 0.25 * randn (m, 1);
%! categor = 2;                  % column 2 is categorical (excludes intercept)
%! S = bootridge (y, X, categor, 100, 0.05, [], 1, 2024);
%! assert (isfield (S, 'Coefficient'));
%! assert (size (S.Coefficient, 1) == 3);
%! assert (isfinite (S.lambda) && (S.lambda > 0));
%! % Check CI bracketing for all coefficients
%! assert (all (S.CI_lower(:) <= S.Coefficient(:) + eps));
%! assert (all (S.CI_upper(:) + eps >= S.Coefficient(:)));
%! assert (S.iter >= 1 && S.expand >= 1);

%!test
%! % Multivariate outcomes and Deff scaling: Sigma_Y_hat should scale by Deff
%! m = 32;
%! x = linspace (-1, 1, m).';
%! X = [ones(m,1), x];
%! B = [2.0, -1.0; 0.5, 0.8];
%! randn ('twister', 123);
%! Y = X * B + 0.2 * randn (m, 2);
%! S1 = bootridge (Y, X, [], 100, 0.10, [], 1, 42);
%! S2 = bootridge (Y, X, [], 100, 0.10, [], 2, 42);
%! assert (all (size (S1.Sigma_Y_hat) == [2, 2]));

