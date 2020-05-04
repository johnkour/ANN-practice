function [J, grad] = NNCostFunction(nn_params, input_layer_size,...
                                    hidden_layer_1_size,...
                                    hidden_layer_2_size,X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs regression.
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.

%=============================INITIAL VALUES===============================

% ---------------TOTAL NUMBER OF ELEMENTS IN THETAS MATRICES---------------

temp_a = hidden_layer_1_size * (input_layer_size + 1);
temp_b = hidden_layer_2_size * (hidden_layer_1_size + 1);
temp_c = 1 * (hidden_layer_2_size + 1);

% -------------------------------------------------------------------------

% Reshape nn_params back into the parameters Theta1, Theta2 and Theta3 the 
% weight matrices for our 3 layer neural network

Theta1 = reshape(nn_params(1:temp_a), hidden_layer_1_size,...
                            (input_layer_size + 1));

Theta2 = reshape(nn_params( (1 + temp_a) : (temp_a + temp_b) ),...
                 hidden_layer_2_size, (hidden_layer_1_size + 1));

Theta3 = reshape(nn_params( (1 + temp_a + temp_b) : end),... 
                            1, (hidden_layer_2_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% We need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Theta3_grad = zeros(size(Theta3));

%==============================MAIN CODE===================================

X = [ones(m,1), X]; % Add the bias elements.

%-----------------------FEEDFORWARD PROPAGATION----------------------------

a_1 = X;
z_2 = a_1*Theta1';
a_2 = sigmoid(z_2);
a_2 = [ones(m,1), a_2];   % Add the bias.
z_3 = a_2*Theta2';
a_3 = sigmoid(z_3);
a_3 = [ones(m,1), a_3];   % Add the bias.
z_4 = a_3*Theta3';
a_4 = sigmoid(z_4);

%--------------------------BACKPROPAGATION---------------------------------

delta_4 = a_4 - y;
delta_3 = delta_4 * Theta3;
delta_3 = delta_3(:,2:end);
delta_2 = delta_3 * Theta2;
delta_2 = delta_2(:,2:end);
delta_2 = delta_2.*sigmoidGradient(z_2);

Theta3_grad = Theta3_grad + delta_4' * a_3;

Theta2_grad = Theta2_grad + delta_3' * a_2;

Theta1_grad = Theta1_grad + delta_2' * a_1;

Theta1_grad = 1/m * Theta1_grad;
Theta1_grad = Theta1_grad + [zeros(size(Theta1_grad,1), 1), lambda/m* Theta1(:,2:end)];

Theta2_grad = 1/m * Theta2_grad;
Theta2_grad = Theta2_grad + [zeros(size(Theta2_grad,1), 1), lambda/m* Theta2(:,2:end)];

Theta3_grad = 1/m * Theta3_grad;
Theta3_grad = Theta3_grad + [zeros(size(Theta3_grad,1), 1), lambda/m* Theta3(:,2:end)];

%-------------------------COST FUNCTION------------------------------------

J = -1/m*sum(sum(y.*log(a_4)+(1-y).*log(1-a_4)));

theta1 = Theta1(:,2:size(Theta1,2));    %We dispose of the bias.
theta2 = Theta2(:,2:size(Theta2,2));    %We dispose of the bias.
theta3 = Theta3(:,2:size(Theta3,2));    %We dispose of the bias.

J = J + lambda/(2*m)*((theta1(:))'*theta1(:)+((theta2(:))'*theta2(:))+((theta3(:))'*theta3(:)));

% --------------------------UNROLL GRADIENTS-------------------------------

grad = [Theta1_grad(:) ; Theta2_grad(:) ; Theta3_grad(:) ];

%================================END=======================================

end


function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

g = 1.0 ./ (1.0 + exp(-z));
end


function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. This should work regardless if z is a matrix or a
%   vector. In particular, if z is a vector or matrix, you should return
%   the gradient for each element.

%=============================INITIAL VALUES===============================

g = zeros(size(z));

%==============================MAIN CODE===================================

g = sigmoid(z) .* (1-sigmoid(z));


%================================END=======================================

end