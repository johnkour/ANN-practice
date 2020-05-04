
%% ========================LOAD AND DIVIDE THE DATA========================

addpath('D:\jkour\Documents\Διπλωματική εργασία\Παλιά εργασία Zeynep',...
        genpath('D:\jkour\Documents\Coursera\Machine Learning\Matlab\machine-learning-ex'));

load('RegressionBlocksSp.mat'); % Open the file with the complete dataset.
X = regression_data(:, 1:2);    y = regression_data(:, 3); % Seperate
% temp = X .^2;                                              % inputs and
% X = [X, temp(:, 1), X(:, 1).* X(:, 2), temp(:, 2)];        % outputs.

clear regression_data;          % Delete original variable, regression_data.
distr = [0.60; 0.20; 0.2];      % Percentages of division of the the total
                                % dataset into 3 subsets, one for training,
                                % one for validation and one for testing.
[X_train, y_train, X_val, y_val, X_test, y_test] = RandomDivision(X, y,...
                                                                    distr);
% Randomly divide the original dataset to three sets according to distr.

% clear temp;

%% ===========================TRAINING ALGORITHM===========================

% ----------------------------GEOMETRY OF THE NN---------------------------

input_layer_size = size(X_train, 2);
hidden_layer_1_size = 3 * input_layer_size; % It is recomended that the
                                            % hidden layer is be 2 to 4
hidden_layer_2_size = 3 * input_layer_size; % times the size of the input
                                            % layer.
                                            
% ---------------------------INITIALIZE THETAS-----------------------------

initial_Theta1 = RandomInitializeWeights(input_layer_size, hidden_layer_1_size);
initial_Theta2 = RandomInitializeWeights(hidden_layer_1_size, hidden_layer_2_size);
initial_Theta3 = RandomInitializeWeights(hidden_layer_2_size, 1);

% -------------------------------------------------------------------------

initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:) ; initial_Theta3(:)];
lambda = 0;

options = optimset('MaxIter', 700);
costFunction = @(p) NNCostFunction(p, input_layer_size,...
                                    hidden_layer_1_size, hidden_layer_2_size,...
                                    X_train, y_train, lambda);

[nn_params, ~] = fminunc(costFunction, initial_nn_params, options);

clear initial_nn_params initial_Theta1 initial_Theta2 initial_Theta3;

% ---------------TOTAL NUMBER OF ELEMENTS IN THETA MATRICES----------------

temp_a = hidden_layer_1_size * (input_layer_size + 1);
temp_b = hidden_layer_2_size * (hidden_layer_1_size + 1);
temp_c = 1 * (hidden_layer_2_size + 1);

% -------------------------------------------------------------------------

Theta1 = reshape(nn_params(1:temp_a), hidden_layer_1_size,...
                            (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + temp_a) : (temp_a + temp_b)),...
                 hidden_layer_2_size, (hidden_layer_1_size + 1));

Theta3 = reshape(nn_params( (1 + temp_a + temp_b) : end),... 
                            1, (hidden_layer_2_size + 1));

clear nn_params temp_a temp_b temp_c;
                        
% -------------------------------------------------------------------------

pred_train = Prediction(Theta1, Theta2, Theta3, X_train);

% -------------------------------------------------------------------------

        immse(pred_train, y_train) * 100

%% ==========================VALIDATION SET================================

% -------------------------------------------------------------------------

pred_val = Prediction(Theta1, Theta2, Theta3, X_val);

% -------------------------------------------------------------------------

        immse(pred_val, y_val) * 100
