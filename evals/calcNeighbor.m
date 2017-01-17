% calculate a logic matrix indicating whether a pair of items are neighbors.
function R = calcNeighbor (dataset, idx1, idx2)

  if ~exist('idx2', 'var')
    idx2 = idx1;
  end

  %T = getConst('MEMORY_CAP');
  T=2000000000;
  
  switch dataset.neighborType

    case 'label'
      N1 = length(idx1);
      N2 = length(idx2);
      R = false(N1, N2);
      m = floor(T / (8 * N1));
      p = 1;
      while p <= N2
        t = min(p + m - 1, N2);
        L1 = dataset.label(idx1);
        L2p = dataset.label(idx2(p: t));
        Dp = repmat(L1, 1, length(L2p)) - repmat(L2p', length(L1), 1);
        R(:, p: t) = Dp == 0;
        p = p + m;
      end

    case 'tag'
	  
      R = dataset.tag(idx1,:) * dataset.tag(idx2,:)'>0;
      

  end

end
