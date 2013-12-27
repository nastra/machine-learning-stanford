function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% compute the cost
prediction = X * theta;
sqrErrors = (prediction-y).^2;
J = 1/(2*m) * sum(sqrErrors);

% regularize theta by removing first value
theta_reg = [0;theta(2:end, :);];

% compute the regularization penalty
penalty = lambda / (2 * m) * theta_reg' * theta;

% compute the overall cost
J = J + penalty;


% =========================================================================

grad = grad(:);
grad = 1 ./ m * X' * (prediction - y) + (lambda/m * theta_reg);

end
