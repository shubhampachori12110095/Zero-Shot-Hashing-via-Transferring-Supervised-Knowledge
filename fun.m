function [F, G] = fun(R, Y, W, B)
  G = 2*(Y*Y'*R - Y*B'*W);
   F = (norm((R'*Y - W'*B),'fro')).^2;
end

