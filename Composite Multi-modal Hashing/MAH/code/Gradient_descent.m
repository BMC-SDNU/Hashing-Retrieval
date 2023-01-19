function [optTheta,functionVal,exitFlag]=Gradient_descent(V, V_thres)  
  
 options = optimset('GradObj','on','MaxIter',100);  
 initialTheta = [0;0];  
 [optTheta,functionVal,exitFlag] = fminunc(@(t)costFunction3(t,V,V_thres),initialTheta,options);  
  
end

function [ jVal,gradient ] = costFunction3( theta, V, V_thres )  
%COSTFUNCTION3 Summary of this function goes here  
%   Logistic Regression  
  
% x=[-3;      -2;     -1;     0;      1;      2;     3];  
% y=[0.01;    0.05;   0.3;    0.45;   0.8;    1.1;    0.99];
x = V;
y = V_thres;
m=size(x,1);  
  
%hypothesis  data  
hypothesis = h_func(x,theta);  
  
% %jVal-cost function  &  gradient updating  
jVal=-sum(sum(log(hypothesis+0.01).*y + (1-y).*log(1-hypothesis+0.01)))/m;
gradient(1)=sum(sum(hypothesis-y))/m - 19;   %reflect to theta1  
gradient(2)=sum(sum((hypothesis-y).*x))/m;    %reflect to theta 2
% jVal = 0;
% gradient(1) = 0;
% gradient(2) = 0;
% for i = 1:size(V,2)
%     jVal = jVal - sum(log(hypothesis(:,i)+0.01).*y(:,i) + (1-y(:,i)).*log(1-hypothesis(:,i)+0.01))/m;
%     gradient(1) = gradient(1) + sum(hypothesis(:,i)-y(:,i))/m;
%     gradient(2) = gradient(2) + sum((hypothesis(:,i)-y(:,i)).*x(:,i))/m;
% end    
  
end  