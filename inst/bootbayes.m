% Performs Bayesian nonparametric bootstrap and calculates posterior statistics 
% for the mean(s) or regression coefficients of univariate or multivariate data.
%
%
% -- Function File: bootbayes (Y)
% -- Function File: bootbayes (Y, X)
% -- Function File: bootbayes (Y, X, CLUSTID)
% -- Function File: bootbayes (Y, X, BLOCKSZ)
% -- Function File: bootbayes (Y, X, ..., NBOOT)
% -- Function File: bootbayes (Y, X, ..., NBOOT, PROB)
% -- Function File: bootbayes (Y, X, ..., NBOOT, PROB, PRIOR)
% -- Function File: bootbayes (Y, X, ..., NBOOT, PROB, PRIOR, SEED)
% -- Function File: bootbayes (Y, X, ..., NBOOT, PROB, PRIOR, SEED, L)
% -- Function File: STATS = bootbayes (Y, ...)
% -- Function File: [STATS, BOOTSTAT] = bootbayes (Y, ...)
%
%     'bootbayes (Y)' performs Bayesian nonparametric bootstrap [1] to create
%     1999 bootstrap statistics, each representing the weighted mean(s) of the
%     column vector (or column-major matrix), Y, using a vector of weights
%     randomly generated from a symmetric Dirichlet distribution. The resulting
%     bootstrap (or posterior [1,2]) distribution(s) is/are summarised for each
%     column of Y (i.e. each outcome) with the following statistics printed to
%     the standard output:
%        - original: the mean(s) of the data column(s) of Y
%        - bias: bootstrap bias estimate(s)
%        - median: the median(s) of the posterior distribution(s)
%        - stdev: the standard deviation(s) of the posterior distribution(s)
%        - CI_lower: lower bound(s) of the 95% credible interval(s)
%        - CI_upper: upper bound(s) of the 95% credible interval(s)
%          By default, the credible intervals are shortest probability
%          intervals, which represent a more computationally stable version
%          of the highest posterior density interval [3].
%
%     'bootbayes (Y, X)' also specifies the design matrix (X) for least squares
%     regression of Y on X. X should be a column vector or matrix the same
%     number of rows as Y. If the X input argument is empty, the default for X
%     is a column of ones (i.e. intercept only) and thus the statistic computed
%     reduces to the mean (as above). The statistics calculated and returned in
%     the output then relate to the coefficients from the regression of Y on X.
%     Y can be a column vector (for univariate regression) or a matrix (for
%     multivariate regression).
%
%     'bootbayes (Y, X, CLUSTID)' specifies a vector or cell array of numbers
%     or strings respectively to be used as cluster labels or identifiers.
%     Rows in Y (and X) with the same CLUSTID value are treated as clusters with
%     dependent errors. Rows of Y (and X) assigned to a particular cluster
%     will have identical weights during Bayesian bootstrap. If empty (default),
%     no clustered resampling is performed and all errors are treated as
%     independent.
%
%     'bootbayes (Y, X, BLOCKSZ)' specifies a scalar, which sets the block size
%     for bootstrapping when the residuals have serial dependence. Identical
%     weights are assigned within each (consecutive) block of length BLOCKSZ
%     during Bayesian bootstrap. Rows of Y (and X) within the same block are
%     treated as having dependent errors. If empty (default), no block
%     resampling is performed and all errors are treated as independent.
%
%     'bootbayes (Y, X, ..., NBOOT)' specifies the number of bootstrap resamples,
%     where NBOOT must be a positive integer. If empty, the default value of
%     NBOOT is 1999.
%
%     'bootbayes (Y, X, ..., NBOOT, PROB)' where PROB is numeric and sets the
%     lower and upper bounds of the credible interval(s). The value(s) of PROB
%     must be between 0 and 1. PROB can either be:
%       <> scalar: To set the central mass of shortest probability intervals
%                  (SPI) to 100*(1-PROB)%
%       <> vector: A pair of probabilities defining the lower and upper
%                  percentiles of the credible interval(s) as 100*(PROB(1))%
%                  and 100*(PROB(2))% respectively. 
%          Credible intervals are not calculated when the value(s) of PROB
%          is/are NaN. The default value of PROB is 0.95.
%
%     'bootbayes (Y, X, ..., NBOOT, PROB, PRIOR)' accepts a positive real
%     numeric scalar to parametrize the form of the symmetric Dirichlet
%     distribution. The Dirichlet distribution is the conjugate PRIOR used to
%     randomly generate weights on the unit simplex for linear least-squares
%     fitting of the observed data, and subsequently to estimate the posterior
%     for the regression coefficients by Bayesian bootstrap and any derived 
%     linear estimates or contrasts.
%
%     If PRIOR is not provided or is empty, the default value of PRIOR is
%     'auto'. The behaviour of 'auto' depends on whether X is provided and
%     whether the model contains slope coefficients.
%
%     If no X is provided, or in intercept-only models, the value 'auto'
%     sets PRIOR so that the Bayesian-bootstrap posterior standard deviation
%     of the mean equals the usual frequentist standard error, i.e. 
%     std(Y,0) / sqrt(N). Here N denotes the number of independent sampling 
%     units (e.g., observations, clusters, or blocks). Thus:
%
%          PRIOR (i.e. alpha) = 1 - 2 / N
%
%     With this setting, std (BOOTSTAT, 0, 2) = std (Y, 0) / sqrt (N) and
%     var (BOOTSTAT, 0, 2) = std (Y, 0)^2 / N (up to Monte Carlo error).
%     When N = 2 (Haldane prior, PRIOR = 0) and the statistic is the mean, the
%     posterior standard deviation equals the frequentist standard error exactly
%     (up to Monte Carlo error):
%
%         std (BOOTSTAT, 1, 2) = std (Y, 1) = std (Y, 0) / sqrt (N)
%
%     If X is a design matrix including slope predictor terms, the value
%     'auto' generalizes the above by providing a global Bessel‑style
%     correction matching the overall variance scale on average across
%     coefficients. Thus:
%
%          PRIOR (i.e. alpha) = 1 - (tr(H) + 1) / N = 1 - (rank(X) + 1) / N
%
%     Here tr(H) (and equivalently rank(X)) is the effective model degrees of
%     freedom, and N is the number of independent sampling units. Equivalently:
%
%          PRIOR (i.e. alpha) = 1 - (N - dfe + 1) / N
%
%     where dfe = N - rank(X) is the effective error degrees of freedom.
%
%     Alternative standard prior choices include: 1 for Bayes’ rule (uniform on
%     the simplex), 0.5 for the transformation-invariant Jeffreys prior (for the
%     Dirichlet weights), and 0 for the Haldane prior. Priors lower than 1 
%     produce a more conservative (wider) posterior, whereas priors greater 
%     than 1 are more liberal, shrinking the posterior bootstrap statistics 
%     toward the maximum-likelihood estimates.
%
%     (For the Haldane prior, normal-quantile CIs use std(BOOTSTAT,1,2) to
%     match the population normalization used for the interval formula.)
%
%     'bootbayes (Y, X, ..., NBOOT, PROB, PRIOR, SEED)' initialises the
%     Mersenne Twister random number generator using an integer SEED value so
%     that 'bootbayes' results are reproducible.
%
%     'bootbayes (Y, X, ..., NBOOT, PROB, PRIOR, SEED, L)' multiplies the
%     regression coefficients by the hypothesis matrix L. If L is not provided
%     or is empty, it will assume the default value of 1 (i.e. no change to
%     the design). Otherwise, L must have the same number of rows as the number
%     of columns in X.
%
%     'STATS = bootbayes (...)' returns a structure where each field contains
%     a matrix of size (p x q), where p is the number of predictors and q is
%     the number of outcomes (columns in Y). Fields include: original, bias, 
%     median, stdev, CI_lower, CI_upper and prior.
%
%     '[STATS, BOOTSTAT] = bootbayes (Y, ...)' also returns the bootstrap
%     statistics. If Y is a column vector, BOOTSTAT is a (p x nboot) matrix. 
%     If Y is a matrix (q > 1), BOOTSTAT is a (1 x q) cell array where each
%     cell contains the (p x nboot) matrix for that outcome.
%
%  Bibliography:
%  [1] Rubin (1981) The Bayesian Bootstrap. Ann. Statist. 9(1):130-134
%  [2] Weng (1989) On a Second-Order Asymptotic property of the Bayesian
%        Bootstrap Mean. Ann. Statist. 17(2):705-710
%  [3] Liu, Gelman & Zheng (2015). Simulation-efficient shortest probability
%        intervals. Statistics and Computing, 25(4), 809–819. 
%
%  bootbayes (version 2026.02.02)
%  Author: Andrew Charles Penn
%  https://www.researchgate.net/profile/Andrew_Penn/
%
%  Copyright 2019 Andrew Charles Penn
%  This program is free software: you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation, either version 3 of the License, or
%  (at your option) any later version.
%
%  This program is distributed in the hope that it will be useful,
%  but WITHOUT ANY WARRANTY; without even the implied warranty of
%  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%  GNU General Public License for more details.
%
%  You should have received a copy of the GNU General Public License
%  along with this program.  If not, see http://www.gnu.org/licenses/


function [stats, bootstat] = bootbayes (Y, X, dep, nboot, prob, prior, seed, ...
                                        L, ISOCTAVE)

  % Check the number of function arguments
  if (nargin < 1)
    error ('bootbayes: Y must be provided')
  end
  if (nargin > 9)
    error ('bootbayes: Too many input arguments')
  end
  if (nargout > 2)
    error ('bootbayes: Too many output arguments')
  end

  % Check if running in Octave (else assume Matlab)
  if (nargin < 9)
    info = ver; 
    ISOCTAVE = any (ismember ({info.Name}, 'Octave'));
  else
    if (~ islogical (ISOCTAVE))
      error ('bootbayes: ISOCTAVE must be a logical scalar.')
    end
  end

  % Calculate the size of Y
  if (nargin < 1)
    error ('bootbayes: DATA must be provided')
  end
  [n, q] = size (Y);

  % Evaluate the design matrix
  if ( (nargin < 2) || (isempty (X)) )
    X = ones (n, 1);
  elseif (size (X, 1) ~= n)
    error ('bootbayes: X must have the same number of rows as y')
  end

  % Remove rows of the data whose outcome or value of any predictor is NaN or Inf
  excl = any ([isnan([Y, X]), isinf([Y, X])], 2);
  Y(excl, :) = [];
  X(excl, :) = [];
  n = n - sum (excl);

  % Calculate the number of parameters
  k = size (X, 2);
  if ( (k == 1) && (all (X == 1)) )
    intercept_only = true;
    p = 1;
    L = 1;
  else
    intercept_only = false;
    % Evaluate hypothesis matrix (L)
    if ( (nargin < 8) || isempty (L) )
      % If L is not provided, set L to 1
      L = 1;
      p = k;
    else
      % Calculate number of parameters
      [m, p] = size (L);
      if (m ~= k)
        error (cat (2, 'bootbayes: the number of rows in L must be the same', ...
                       ' as the number of columns in X'))
      end
    end
  end

  % Check for missing data
  if (any (any ([isnan(X), isinf(X)], 2)))
    error ('bootbayes: elements of X cannot be NaN or Inf')
  end
  if (~ intercept_only)
    if (any (any ([isnan(Y), isinf(Y)], 2)))
      error (cat (2, 'bootbayes: elements of y cannot be NaN or Inf if the', ... 
                     ' model is not an intercept-only model'))
    end
  end

  % Evaluate cluster IDs or block size
  if ( (nargin > 2) && (~ isempty (dep)) )
    if (isscalar (dep))
      % Prepare for block Bayesian bootstrap
      blocksz = dep;
      N = fix (n / blocksz);
      IC = (N + 1) * ones (n, 1);
      IC(1:blocksz * N, :) = reshape (ones (blocksz, 1) * (1:N), [], 1);
      N = IC(end);
      method = 'block ';
    else
      % Prepare for cluster Bayesian bootstrap
      dep(excl) = [];
      clustid = dep;
      if ( any (size (clustid) ~= [n, 1]) )
        error (cat (2, 'bootbayes: CLUSTID must be a column vector with', ...
                       ' the same number of rows as Y'))
      end
      [C, jnk, IC] = unique (clustid);
      N = numel (C); % Number of clusters
      method = 'cluster ';
    end
  else
    N = n;
    IC = [];
    method = '';
  end
  if (N < 2)
    error ('bootbayes: Y must contain more than one independent sampling unit')
  end

  % Evaluate number of bootstrap resamples
  if ( (nargin < 4) || (isempty (nboot)) )
    nboot = 1999;
  else
    if (~ isa (nboot, 'numeric'))
      error ('bootbayes: NBOOT must be numeric')
    end
    if (numel (nboot) > 1)
      error ('bootbayes: NBOOT must be scalar')
    end
    if (nboot ~= abs (fix (nboot)))
      error ('bootbayes: NBOOT must be a positive integer')
    end
  end

  % Evaluate prob
  if ( (nargin < 5) || (isempty (prob)) )
    prob = 0.95;
    nprob = 1;
  else
    nprob = numel (prob);
    if (~ isa (prob, 'numeric') || (nprob > 2))
      error ('bootbayes: PROB must be a scalar or a vector of length 2')
    end
    if (size (prob, 1) > 1)
      prob = prob.';
    end
    if (any ((prob < 0) | (prob > 1)))
      error ('bootbayes: Value(s) in PROB must be between 0 and 1')
    end
    if (nprob > 1)
      % PROB is a pair of probabilities
      % Make sure probabilities are in the correct order
      if (prob(1) > prob(2) )
        error (cat (2, 'bootbayes: The pair of probabilities must be in', ...
                       ' ascending numeric order'))
      end
    end
  end

  % Evaluate or set prior
  if ( (nargin < 6) || (isempty (prior)) )
    prior = 'auto';
  end
  if (~ isa (prior, 'numeric'))
    if (strcmpi (prior, 'auto'))
      % Automatic prior selection to produce a posterior whose variance is an
      % unbiased estimator of the sampling variance (on average in the case of
      % regression models)
      if (intercept_only)
        prior = 1 - 2 / N;
      else
        % If X is full rank, rank (X) = k
        prior = max (0, 1 - (rank (X) + 1) / N); 
      end
    else
      error ('bootbayes: PRIOR must be numeric or ''auto''');
    end
  end
  if (numel (prior) > 1)
    error ('bootbayes: PRIOR must be scalar');
  end
  if any (prior ~= abs (prior))
    error ('bootbayes: PRIOR must be non-negative (>= 0)');
  end

  % Set random seed
  if ( (nargin > 6) && (~ isempty (seed)) )
    if (ISOCTAVE)
      rand ('seed', seed);
      randg ('seed', seed);
    else
      rand ('seed', seed);
      randn ('seed', seed);
    end
  end

  % Create weights by randomly sampling from a symmetric Dirichlet distribution.
  % This can be achieved by normalizing a set of randomly generated values from
  % a Gamma distribution to their sum.
  if (prior > 0)
    if (ISOCTAVE)
      r = randg (prior, N, nboot);
    else
      if ((exist ('gammaincinv', 'builtin')) || ...
          (exist ('gammaincinv', 'file')))
        r = gammaincinv (rand (N, nboot), prior); % Fast
      else
        % Earlier versions of Matlab do not have gammaincinv
        % Instead, use functions from the Statistics and Machine Learning Toolbox
        try 
          r = gaminv (rand (N, nboot), prior, 1); % Fast
        catch
          r = gamrnd (prior, 1, N, nboot); % Slow 
        end
      end
    end
  else
    % Haldane prior
    r = zeros (N, nboot);
    idx = fix (rand (1, nboot) * N + (1:N:(nboot * N)));
    r(idx)=1;
  end
  if (~ isempty (IC))
    r = r(IC, :);  % Enforce clustering/blocking
  end
  W = bsxfun (@rdivide, r, sum (r));

  % Compute bootstap statistics
  if (intercept_only)
    bootfun = @(Y) sum (bsxfun (@times, Y, W));  % Faster!
    original = mean (Y, 1);
    bootstat = arrayfun (@(j) bootfun (Y(:, j)), 1:q, 'UniformOutput', false);
  else
    bootfun = @(w) lmfit (X, Y, w, L);
    original = bootfun (ones (n, 1) / n);
    W = sqrt (W);  % Square root of the weights for regression.
    bootstat = cell2mat (arrayfun (@(b) bootfun (W(:, b)), 1:nboot, ...
                                   'UniformOutput', false));
    bootstat = arrayfun (@(i) bootstat(:, i:q:end), 1:q, 'UniformOutput', false);
  end

  % Initialize output structure
  stats = struct;
  stats.original = original;

  % Bootstrap bias estimation
  stats.bias = cell2mat (arrayfun (@(j) mean (bootstat{j}, 2) - ...
                         original(:, j), 1:q, 'UniformOutput', false));

  % Central tendency of the bootstrap distribution
  stats.median = cell2mat (arrayfun (@(j) median (bootstat{j}, 2), 1:q, ...
                           'UniformOutput', false));

  % Scale of the bootstrap distribution
  stats.stdev = cell2mat (arrayfun (@(j) std (bootstat{j}, 1, 2), 1:q, ...
                          'UniformOutput', false));

  % Compute credible intervals
  CI_lower = nan (p, q);
  CI_upper = nan (p, q);
  if (any (~ isnan (prob)))
    if (prior > 0)
      for j = 1:q
        ci = credint (bootstat{j}, prob);
        CI_lower(:, j) = ci(:, 1); CI_upper(:, j) = ci(:, 2);
      end
    else
      stdnorminv = @(p) sqrt (2) * erfinv (2 * p - 1);
      switch nprob
        case 1
          z = stdnorminv (1 - (1 - prob) / 2);
          for j = 1:q
            sd = std (bootstat{j}, 1 , 2);
            ci = bsxfun (@plus, original(:, j), 
                         bsxfun (@times, [-1, 1], sd * z));
            CI_lower(:, j) = ci(:, 1); CI_upper(:, j) = ci(:, 2);
          end
        case 2
          z = stdnorminv (prob);
          for j = 1:q
            sd = std (bootstat{j}, 1 , 2);
            ci = bsxfun (@plus, original(:, j), bsxfun (@times, sd, z));
            CI_lower(:, j) = ci(:, 1); CI_upper(:, j) = ci(:, 2);
          end
      end
    end
  end
  stats.CI_lower = CI_lower;
  stats.CI_upper = CI_upper;

  % Attach the prior to the output structure
  stats.prior = prior;

  % If Y is a single outcome, return bootstat as a matrix
  if (q == 1)
    bootstat = cell2mat (bootstat);
  end

  % Print output if no output arguments are requested
  if (nargout == 0) 
    print_out (stats, nboot, prob, prior, p, q, L, method, intercept_only);
  end

end

%--------------------------------------------------------------------------

% FUNCTION TO FIT THE LINEAR MODEL

function param = lmfit (X, y, w, L)

  % Get model coefficients by solving the linear equation.
  % Solve the linear least squares problem using the Moore-Penrose pseudo
  % inverse (pinv) to make it more robust to the situation where X is singular.
  % If optional arument w is provided, it should be equal to square root of the
  % weights we want to apply to the regression.
  n = numel (y);
  if ( (nargin < 3) || isempty (w) )
    % If no weights are provided, create a vector of ones
    w = ones (n, 1);
  end
  if ( (nargin < 4) || isempty (L) )
    % If no hypothesis matrix (L) is provided, set L to 1
    L = 1;
  end

  % Solve the linear equation to minimize weighted least squares, where the
  % weights are equal to w.^2. The whitening transformation actually implemented
  % avoids using an n * n matrix and has improved accuracy over the solution
  % using the equivalent normal equation:
  %   b = pinv (X' * W * X) * (X' * W * y);
  % Where W is the diagonal matrix of weights (i.e. W = diag (w.^2))
  b = pinv (bsxfun (@times, w, X)) * bsxfun (@times, w, y);
  param = L' * b;

end

%--------------------------------------------------------------------------

% FUNCTION TO PRINT OUTPUT

function print_out (stats, nboot, prob, prior, p, q, L, method, intercept_only)

    fprintf (cat (2, '\nSummary of Bayesian bootstrap estimates of bias', ...
                     ' and precision for linear models\n', ...
                     '*************************************************', ...
                     '******************************\n\n'));
    fprintf ('Bootstrap settings: \n');
    if (intercept_only)
        fprintf (' Function: sum (W * Y)\n');
    else
      if ( (numel(L) > 1) || (L ~= 1) )
        fprintf (' Function: L'' * pinv (X'' * W * X) * (X'' * W * y)\n');
      else
        fprintf (' Function: pinv (X'' * W * X) * (X'' * W * y)\n');
      end
    end
    fprintf (' Resampling method: Bayesian %sbootstrap\n', method)
    fprintf (' Prior: Symmetric Dirichlet distribution (alpha = %.3g)\n', prior)
    fprintf (' Number of resamples: %u \n', nboot)
    if (~ isempty (prob) && ~ all (isnan (prob)))
      nprob = numel (prob);
      if (nprob > 1)
        % prob is a vector of probabilities
        fprintf (' Credible interval (CI) type: Percentile interval\n');
        mass = 100 * abs (prob(2) - prob(1));
        fprintf (' Credible interval: %.3g%% (%.1f%%, %.1f%%)\n', ...
                 mass, 100 * prob);
      else
        % prob is a two-tailed probability
        fprintf (cat (2, ' Credible interval (CI) type: Shortest', ...
                         ' probability interval\n'));
        mass = 100 * prob;
        fprintf (' Credible interval: %.3g%%\n', mass);
      end
    end
    for j = 1:q
      fprintf ('\nPosterior statistics for outcome %d: \n', j);
      fprintf (cat (2, ' original     bias         median       stdev', ... 
                       '       CI_lower      CI_upper\n'));
      for i = 1:p
        fprintf (cat (2, ' %#-+10.4g   %#-+10.4g   %#-+10.4g   %#-10.4g', ...
                         '  %#-+10.4g    %#-+10.4g\n'), ... 
                 [stats.original(i, j), stats.bias(i, j), stats.median(i, j), ...
                  stats.stdev(i, j), stats.CI_lower(i, j), stats.CI_upper(i, j)]);
      end
    end
    fprintf ('\n');

end

%--------------------------------------------------------------------------

%!demo
%!
%! % Input univariate dataset
%! heights = [183, 192, 182, 183, 177, 185, 188, 188, 182, 185].';
%!
%! % 95% credible interval for the mean 
%! bootbayes (heights);
%!
%! % Please be patient, the calculations will be completed soon...

%!demo
%!
%! % Input bivariate dataset
%! X = [ones(43,1),...
%!     [01,02,03,04,05,06,07,08,09,10,11,...
%!      12,13,14,15,16,17,18,19,20,21,22,...
%!      23,25,26,27,28,29,30,31,32,33,34,...
%!      35,36,37,38,39,40,41,42,43,44]'];
%! y = [188.0,170.0,189.0,163.0,183.0,171.0,185.0,168.0,173.0,183.0,173.0,...
%!     173.0,175.0,178.0,183.0,192.4,178.0,173.0,174.0,183.0,188.0,180.0,...
%!     168.0,170.0,178.0,182.0,180.0,183.0,178.0,182.0,188.0,175.0,179.0,...
%!     183.0,192.0,182.0,183.0,177.0,185.0,188.0,188.0,182.0,185.0]';
%!
%! % 95% credible interval for the regression coefficents
%! bootbayes (y, X);
%!
%! % Please be patient, the calculations will be completed soon...

%!demo
%! ## --- Stress-test: Simulated Large-Scale Patch-seq Project (bootbayes) ---
%! ## N = 7500 cells (observations), p = 15 features, q = 2000 genes (outcomes).
%! ## This tests multivariate un-weaving logic and HPD interval scaling.
%!
%! N = 7500;       
%! p = 15;         
%! q = 2000;       
%! nboot = 500;    % Sufficient for stable HPD intervals
%!
%! printf('Simulating a Large-Scale Patch-seq Dataset (%d x %d)...\n', N, q);
%!
%! % Generate design matrix X (e.g., E-phys features)
%! X = [ones(N,1), randn(N, p-1)];
%!
%! % Generate multivariate outcome Y (Gene expression)
%! % Approx 120MB of data
%! true_beta = randn(p, q) .* (rand(p, q) > 0.9); 
%! Y = X * true_beta + randn(N, q) * 0.5;
%!
%! printf('Running bootbayes with %d bootstrap resamples...\n', nboot);
%! tic;
%! % Running with default alpha=0.05 (95% Credible Intervals)
%! stats = bootbayes(Y, X, [], nboot);
%! runtime = toc;
%!
%! printf('\n--- Performance Results ---\n');
%! printf('Runtime: %.2f seconds\n', runtime);
%! printf('Total parameters estimated: %d\n', p * q);
%!
%! % Accuracy Check on a random gene
%! target_gene = randi(q);
%! % Note: checking if original is returned as p x q matrix
%! if all(size(stats.original) == [p, q])
%!    estimated = stats.original(:, target_gene);
%! else
%!    % Fallback if it is returned as the interleaved vector
%!    estimated = stats.original((target_gene-1)*p + (1:p));
%! endif
%!
%! actual = true_beta(:, target_gene);
%! correlation = corr(estimated, actual);
%!
%! printf('Correlation of estimates for Gene %d: %.4f\n', target_gene, correlation);
%!
%! % Verify Credible Interval structure
%! printf('Number of outcome cells in stats.stdev: %d\n', length(stats.stdev));
%!
%! if correlation > 0.98
%!   printf('Result: PASSED (High accuracy OLS recovery)\n');
%! else
%!   printf('Result: WARNING (Low correlation - check for colinearity)\n');
%! endif

%!test
%! % Test calculations of statistics for the mean
%!
%! % Input univariate dataset
%! heights = [183, 192, 182, 183, 177, 185, 188, 188, 182, 185].';
%!
%! % 95% credible interval for the mean 
%! stats = bootbayes(heights);
%! stats = bootbayes(repmat(heights,1,5));
%! stats = bootbayes(heights,ones(10,1));
%! stats = bootbayes(heights,[],2);
%! stats = bootbayes(heights,[],[1;1;1;1;2;2;2;3;3;3]);
%! stats = bootbayes(heights,[],[],1999);
%! stats = bootbayes(heights,[],[],[],0.05);
%! stats = bootbayes(heights,[],[],[],[0.025,0.975]);
%! stats = bootbayes(heights,[],[],[],[]);
%! stats = bootbayes(heights,[],[],[],[],[],[]);
%! [stats,bootstat] = bootbayes(heights);

%!test
%! % Test calculations of statistics for linear regression
%!
%! % Input bivariate dataset
%! X = [ones(43,1),...
%!     [01,02,03,04,05,06,07,08,09,10,11,...
%!      12,13,14,15,16,17,18,19,20,21,22,...
%!      23,25,26,27,28,29,30,31,32,33,34,...
%!      35,36,37,38,39,40,41,42,43,44]'];
%! y = [188.0,170.0,189.0,163.0,183.0,171.0,185.0,168.0,173.0,183.0,173.0,...
%!     173.0,175.0,178.0,183.0,192.4,178.0,173.0,174.0,183.0,188.0,180.0,...
%!     168.0,170.0,178.0,182.0,180.0,183.0,178.0,182.0,188.0,175.0,179.0,...
%!     183.0,192.0,182.0,183.0,177.0,185.0,188.0,188.0,182.0,185.0]';
%!
%! % 95% credible interval for the mean 
%! stats = bootbayes(y,X);
%! stats = bootbayes(y,X,4);
%! stats = bootbayes(y,X,[],1999);
%! stats = bootbayes(y,X,[],[],0.05);
%! stats = bootbayes(y,X,[],[],[0.025,0.975]);
%! stats = bootbayes(y,X,[],[]);
%! [stats,bootstat] = bootbayes(y,X);
