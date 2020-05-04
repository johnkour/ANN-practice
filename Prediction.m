function p = Prediction(Theta1, Theta2, Theta3, X)
%PREDICT Predict the value of an input given a trained neural network.
%   p = PREDICT(Theta1, Theta2, Theta3, X) outputs the predicted value of X
%   given the trained weights of a neural network (Theta1, Theta2, Theta3).

%=============================INITIAL VALUES===============================

m = size(X, 1);
p = zeros(m, 1);
X = [ones(m,1), X]; % Add the bias elements.

%==============================MAIN CODE===================================

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

%------------------------------PREDICTION----------------------------------

p = a_4;

%================================END=======================================

end


function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

g = 1.0 ./ (1.0 + exp(-z));
end