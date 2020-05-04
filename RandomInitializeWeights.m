function W = RandomInitializeWeights(L_in, L_out)
%RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
%incoming connections and L_out outgoing connections
%   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights 
%   of a layer with L_in incoming connections and L_out outgoing 
%   connections. 
%
%   Note that W should be set to a matrix of size(L_out, 1 + L_in) as
%   the first column of W handles the "bias" terms

%=============================INITIAL VALUES===============================

W = zeros(L_out, 1 + L_in);

EPSILON = sqrt( 6/(L_in+L_out+1) ); %Recomended value of epsilon.

%==============================MAIN CODE===================================

W = rand(L_out, L_in+1) * 2*EPSILON - EPSILON;

%================================END=======================================

end