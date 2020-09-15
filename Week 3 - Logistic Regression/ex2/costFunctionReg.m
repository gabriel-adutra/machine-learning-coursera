function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
n = size(theta);
htheta1 = X*theta;
htheta = sigmoid(htheta1);
a = (1-2)*y.*log(htheta);
b = (1-y).*log(1-htheta);
delta = a - b; 
matrixJ = (delta)/(m);
P=sum(matrixJ)
matrixK = theta(2:n).*theta(2:n).*lambda/(2*m);
G= sum(matrixK);
J= P + G; 


A = htheta - y
grad = 1/m*(X'*A)+lambda/m*theta;
grad(1) = grad(1) - lambda*theta(1)/m;



% =============================================================

end
