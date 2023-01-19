function [res] = h_func(inputx,theta)  
  
%cost function 3  
tmp=theta(1)+theta(2)*inputx;%m*1  
res=1./(1+exp(-tmp));%m*1  
  
end 




  