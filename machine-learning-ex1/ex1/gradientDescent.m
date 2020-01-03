function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for i = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
     temp = zeros(length(theta),1);
     for j = 1:length(temp)
           temp(j) = theta(j) - alpha * (1 / m) * sum(((X * theta) - y) .* X(:, j));
     endfor
     for j = 1:length(temp)
       theta(j)=temp(j);
     endfor

    % ============================================================

    % Save the cost J in every iteration    
    J_history(i) = computeCost(X, y, theta);

end

end
