% This function returns resampled data or indices created by balanced bootstrap 
% or bootknife resampling.
%
% -- Function File: BOOTSAM = boot (N, NBOOT)
% -- Function File: BOOTSAM = boot (X, NBOOT)
% -- Function File: BOOTSAM = boot (..., NBOOT, LOO)
% -- Function File: BOOTSAM = boot (..., NBOOT, LOO, SEED)
% -- Function File: BOOTSAM = boot (..., NBOOT, LOO, SEED, WEIGHTS)
%
%     'BOOTSAM = boot (N, NBOOT)' generates NBOOT bootstrap samples of length N.
%     The samples generated are composed of indices within the range 1:N, which
%     are chosen by random resampling with replacement [1]. N and NBOOT must be
%     positive integers. The returned value, BOOTSAM, is a matrix of indices,
%     with N rows and NBOOT columns. The efficiency of the bootstrap simulation
%     is ensured by sampling each of the indices exactly NBOOT times, for first-
%     order balance [2-3]. Balanced resampling only applies when NBOOT > 1.
%
%     'BOOTSAM = boot (X, NBOOT)' generates NBOOT bootstrap samples, each the
%     same length as X (N). X must be a numeric vector, and NBOOT must be
%     positive integer. BOOTSAM is a matrix of values from X, with N rows
%     and NBOOT columns. The samples generated contains values of X, which
%     are chosen by balanced bootstrap resampling as described above [1-3].
%     Balanced resampling only applies when NBOOT > 1.
%
%     Note that the values of N and NBOOT map onto int32 data types in the 
%     boot MEX file. Therefore, these values must never exceed (2^31)-1.
%
%     'BOOTSAM = boot (..., NBOOT, LOO)' sets the resampling method. If LOO
%     is false, the resampling method used is balanced bootstrap resampling.
%     If LOO is true, the resampling method used is balanced bootknife
%     resampling [4]. The latter involves creating leave-one-out (jackknife)
%     samples of size N - 1, and then drawing resamples of size N with
%     replacement from the jackknife samples, thereby incorporating Bessel's
%     correction into the resampling procedure. LOO must be a scalar logical
%     value. The default value of LOO is false.
%
%     'BOOTSAM = boot (..., NBOOT, LOO, SEED)' sets a seed to initialize
%     the pseudo-random number generator to make resampling reproducible between
%     calls to the boot function. Note that the mex function compiled from the
%     source code boot.cpp is not thread-safe. Below is an example of a line of
%     code one can run in Octave/Matlab before attempting parallel operation of
%     boot.mex in order to ensure that the initial random seeds of each thread
%     are unique:
%       • In Octave:
%            pararrayfun (nproc, @boot, 1, 1, false, 1:nproc)
%       • In Matlab:
%            ncpus = feature('numcores'); 
%            parfor i = 1:ncpus; boot (1, 1, false, i); end;
%
%     'BOOTSAM = boot (..., NBOOT, LOO, SEED, WEIGHTS)' sets a weight
%     vector of length N. If WEIGHTS is empty or not provided, the default 
%     is a vector of length N, with each element equal to NBOOT (i.e. uniform
%     weighting). Each element of WEIGHTS is the number of times that the
%     corresponding index (or element in X) is represented in BOOTSAM.
%     Therefore, the sum of WEIGHTS must equal N * NBOOT. 
%
%  Bibliography:
%  [1] Efron, and Tibshirani (1993) An Introduction to the
%        Bootstrap. New York, NY: Chapman & Hall
%  [2] Davison et al. (1986) Efficient Bootstrap Simulation.
%        Biometrika, 73: 555-66
%  [3] Booth, Hall and Wood (1993) Balanced Importance Resampling
%        for the Bootstrap. The Annals of Statistics. 21(1):286-298
%  [4] Hesterberg T.C. (2004) Unbiasing the Bootstrap—Bootknife Sampling 
%        vs. Smoothing; Proceedings of the Section on Statistics & the 
%        Environment. Alexandria, VA: American Statistical Association.
%
%  boot (version 2024.04.24)
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

function bootsam = boot (x, nboot, loo, s, w)

  % Input variables
  n = numel(x);
  if (n > 1)
    sz = size (x);
    isvec = true;
    if (all (sz > 1))
      error (cat (2, 'boot: The first input argument must be either a', ...
                     ' scalar (N) or vector (X).'))
    end
  else
    n = x;
    isvec = false;
    if ((n <= 0) || (n ~= fix (n)) || isinf (n) || isnan (n))
      error (cat (2, 'boot: The first input argument must be a finite', ...
                     ' positive integer.'))
    end
  end
  if ((nboot <= 0) || (nboot ~= fix (nboot)) || isinf (nboot) || ...
       isnan (nboot) || (max (size (nboot)) > 1))
    error (cat (2, 'boot: The second input argument (NBOOT) must be a', ...
                   ' finite positive integer'))
  end
  if ((nargin > 2) && ~ isempty (loo))
    if ( (~ isscalar (loo)) || (~ islogical (loo)) )
      error (cat (2, 'boot: The third input argument (LOO) must be', ...
                     ' a logical scalar value'))
    end
  else
    loo = false;
  end
  if ((nargin > 3) && ~ isempty (s))
    if (isinf (s) || isnan (s) || (max (size (s)) > 1))
      error (cat (2, 'boot: The fourth input argument (SEED) must be a', ...
                     ' finite scalar value'))
    end
    rand ('twister', s);
  end

  % Preallocate bootsam
  bootsam = zeros (n, nboot);

  % Initialize weight vector defining the available row counts remaining
  if ((nargin > 4) && ~ isempty (w))
    % Assign user defined weights (counts)
    % Error checking
    if (numel (w) ~= n)
      error (cat (2, 'boot: WEIGHTS must be a vector of length N or be', ...
                     ' the same length as X.'))
    end
    if (sum (w) ~= n * nboot)
      error ('boot: The elements of WEIGHTS must sum to N * NBOOT.')
    end
    c = w(:);
  else
    % Assign weights (counts) for uniform sampling
    c = ones (n, 1) * nboot; 
  end

  % Initialize
  N = sum (c);
  vectorized = true;
  if (loo)
    % If bootknife resampling, create a vector (r) containing the index of the 
    % observation to omit from each of the nboot resamples
    b = (1 : nboot);
    r = b - fix ((b - 1) / n) * n;                        % systematic
    nr = sum ((fix ((b - 1) / n) == fix (nboot / n)));
    r(end - nr + 1 : end)  = 1 + fix (rand (1, nr) * n);  % random
  end

  % Perform balanced sampling
  for b = 1:nboot

    % Create n pseudo-random numbers for resample number b
    R = rand (1, n);

    % Re-evaluate whether to use vectorized resampling. The resampling is only
    % vectorized when the count (c) for each of the indices is >= n
    if (vectorized)
      if (any (c < n))
        vectorized = false;
      end
    end

    % Additional steps relevant to bootknife resampling only
    if (loo)
      % Only leave-one-out if omitting sample index r leaves greater than or
      % equal to n sample counts summed across the other indices (unless we are
      % only requesting a single random bootstrap resample)
      if ((N - c(r(b)) >= n) || (nboot == 1))
        m = c(r(b));
        c(r(b)) = 0;
      else
        loo = false;
      end
    end

    % Generate resampled indices using the pseudo-random numbers
    if (vectorized)
      % Vectorized resampling - faster algorithm that can be used to generate
      % most of the resamples. It is approximate since weights/counts only
      % update after each data set is resampled. 
      d = cumsum (c);
      j = sum (bsxfun (@ge, R * d(n), d)) + 1;
      if (nboot > 1)
        c = c - sum (bsxfun (@eq, j, (1 : n)'), 2);
        N = N - n;
      end
    else
      % Non-vectorized resampling - slower but guarantees exact first order
      % balance as we approach nboot resamples. Weights/counts update after
      % each observation is sampled.
      j = zeros (n, 1);
      for i = 1:n
        d = cumsum (c);
        j(i) = sum (R(i) * d(n) >= d) + 1;
        if (nboot > 1)
          c(j(i)) = c(j(i)) - 1;
          N = N - 1;
        end
      end
    end

    % Assign resampled data or indices to output
    if (isvec) 
      bootsam (:, b) = x(j);
    else
      bootsam (:, b) = j;
    end

    % Additional steps relevant to bootknife resampling only
    if (loo)
      c(r(b)) = m;
      m = 0;
    end

  end

%!demo
%!
%! % N as input; balanced bootstrap resampling with replacement
%! boot(3, 20, false)

%!demo
%!
%! % N as input; (unbiased) balanced bootknife resampling with replacement
%! boot(3, 20, true)

%!demo
%! 
%! % N as input; balanced resampling with replacement; setting the random seed
%! boot(3, 20, false, 1) % Set random seed
%! boot(3, 20, false, 1) % Reset random seed, BOOTSAM is the same
%! boot(3, 20, false)    % Without setting random seed, BOOTSAM is different

%!demo
%!
%! % Vector (X) as input; balanced resampling with replacement; setting weights
%! x = [23; 44; 36];
%! boot(x, 10, false, 1)            % equal weighting
%! boot(x, 10, false, 1, [20;0;10]) % unequal weighting, no x(2) in BOOTSAM 

%!demo
%!
%! % N as input; resampling without replacement; 3 trials
%! boot(6, 1, false, 1) % Sample 1; Set random seed for first sample only
%! boot(6, 1, false)    % Sample 2
%! boot(6, 1, false)    % Sample 3

%!test
%! % Test that random samples vary between calls to boot.
%! I1 = boot (3, 20);
%! I2 = boot (3, 20);
%! assert (all (I1(:) == I2(:)), false);

%!test
%! % Test that random seed gives identical resamples when LOO is false.
%! I1 = boot (3, 20, false, 1);
%! I2 = boot (3, 20, false, 1);
%! assert (all (I1(:) == I2(:)), true);

%!test
%! % Test that random seed gives identical resamples when LOO is true.
%! I1 = boot (3, 20, true, 1);
%! I2 = boot (3, 20, true, 1);
%! assert (all (I1(:) == I2(:)), true);

%!test
%! % Test that default setting for LOO is false.
%! I1 = boot (3, 20, [], 1);
%! I2 = boot (3, 20, false, 1);
%! assert (all (I1(:) == I2(:)), true);

%!test
%! % Test that resampling is balanced when LOO is false.
%! I = boot (3, 20, false, 1);
%! assert (sum (I(:) == 1), 20, 1e-03);
%! assert (sum (I(:) == 2), 20, 1e-03);
%! assert (sum (I(:) == 3), 20, 1e-03);

%!test
%! % Test that resampling is balanced when LOO is true.
%! I = boot (3, 20, true, 1);
%! assert (sum (I(:) == 1), 20, 1e-03);
%! assert (sum (I(:) == 2), 20, 1e-03);
%! assert (sum (I(:) == 3), 20, 1e-03);

%! % Test for unbiased sampling (except for last sample).
%! % The exception is a requirement to maintain balance.
%! I = boot (3, 20, true, 1);
%! assert (all (diff (sort (I(1:end-1)))), false);

%!test
%! % Test feature for changing resampling weights when LOO is false
%! I = boot (3, 20, false, 1, [30,30,0]);
%! assert (any (I(:) == 3), false);

%!test
%! % Test feature for changing resampling weights when LOO is true
%! I = boot (3, 20, true, 1, [30,30,0]);
%! assert (any (I(:) == 3), false);
