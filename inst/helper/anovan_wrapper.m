function  F = anovan_wrapper (y, g, ISOCTAVE, options)
  
  % Helper function file required for bootanovan
  
  % anovan_wrapper cannot be a subfunction or nested function since 
  % Octave parallel threads won't be able to find it
  
  if ISOCTAVE
    % Octave anovan
    [junk,F] = anovan(y,g,options{:});
  else
    % Matlab anovan
    [junk,tbl] = anovan(y,g,'display','off',options{:});
    F = cell2mat(tbl(2:end,6)); 
  end

end