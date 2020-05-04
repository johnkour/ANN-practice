function [X_train,y_train, X_val, y_val, X_test, y_test] = RandomDivision(X,y, distr)
%	Uses the three elements of the vector distr to divide randomly the
%	initial dataset to three subsets: one for training, one for validation
%	and one for testing.
%   	X is the initial input matrix, y is the initial output vector and
%   	distr is the 3x1 vector that contains the factors for the division.
%   	X_train, X_val, X_test are the matices that will be used for
%   	training, validation and testing by our supervised learning
%   	algorithm.

%=============================INITIAL VALUES===============================

[m, n] = size(X);   % m:training examples, n: variables/parameters.

%========================AUXILIARY CALCULATIONS============================

%-------------------------------INDEXING-----------------------------------

ind = randperm(m);  % ind: m training examples in random order.

m_train = floor(distr(1)*m);    % m_train: number of training examples.
m_val = floor(distr(2)*m);      % m_val: number of examples for validation.
% m_test = m - (m_train + m_val); % m_test: number of testing examples.
%   By the previus calculations m_train and m_val may be a little less than
%   distr(1)*m and distr(2)*m, but the test set will be slightly bigger
%   than distr(3)*m. This allows us to test our hypothesis better.

ind_train = ind(1:m_train);              % Indexing for the training set.
ind_val = ind(m_train+1: m_train+m_val); % Indexing for the validation set.
ind_test = ind(m_train+m_val+1: end);    % Indexing for the test set.

%-----------------------------NEW MATRICES---------------------------------

X_train = X(ind_train, :); y_train = y(ind_train);
X_val = X(ind_val, :);     y_val = y(ind_val);
X_test = X(ind_test, :);   y_test = y(ind_test);

%===================================END====================================

end