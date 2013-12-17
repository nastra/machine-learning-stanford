function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for i=1:num_iters
 cost = X' * (X * theta - y);
 theta = theta - ((alpha/m) * cost);
 % Save the cost J in every iteration
 J_history(i) = computeCostMulti(X, y, theta);
 end

end
