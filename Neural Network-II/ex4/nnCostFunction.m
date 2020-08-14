function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

a1=[ones(rows(X), 1), X];
z2=a1*Theta1';
a2=sigmoid(z2);
a2=[ones(rows(a2), 1), a2];
z3=a2*Theta2';
h=sigmoid(z3);%a3 = h(X)

%Implementing feedforward propagation to compute cost function

J=zeros(num_labels,1);
for i=1:num_labels
  temp_y=(y == i);
  J(i)=sum((-temp_y'*log(h(1:rows(h),i)))+(-(1-temp_y)'*log(1-h(1:rows(h),i))));
endfor
J=(1/m)*sum(J);
temp1 = temp2 = 0;
for i = 1:rows(Theta1)
  temp1 = temp1 + sum(Theta1(i,2:end).^2);
endfor
for i = 1:rows(Theta2)
  temp2 = temp2 + sum(Theta2(i,2:end).^2);
endfor
J=J+((lambda/(2*m))*(temp1+temp2)); 

%Implementing backpropagation algorithm for computing partial gradients

tri1=tri2=0;

for t=1:m
  a1=[1, X(t,:)];
  z2=a1*Theta1';
  a2=sigmoid(z2);
  a2=[1, a2];
  z3=a2*Theta2';
  h=sigmoid(z3);%a3 = h(X)
  del3=zeros(num_labels,1);
  del2=0;
  %Initializing del3 and computing del2
  for i=1:num_labels
    del3(i)=(h(i)'-(y(t)==i));
  endfor
  del2=((Theta2(1:end,2:end))'*del3).*sigmoidGradient(z2');
  
  %Accumulating the gradients
  
  tri1=tri1+(del2*a1);
  tri2=tri2+(del3*a2);
endfor

%Computing Theta1_grad and Theta2_grad

  Theta1_grad=(1/m)*tri1;
  Theta2_grad=(1/m)*tri2;

%Regularizing the gradient values

for i = 2:columns(Theta1)
  Theta1_grad(1:end,i)=Theta1_grad(1:end,i)+((lambda/m)*Theta1(1:end,i));
endfor

for j=2:columns(Theta2)
  Theta2_grad(1:end,j)=Theta2_grad(1:end,j)+((lambda/m)*Theta2(1:end,j));
endfor

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
