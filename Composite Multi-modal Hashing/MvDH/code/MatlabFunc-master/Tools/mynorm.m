function V = mynorm(M,dim)
%   MYNORM: �������ĳ��ά���ϵ�������2����
%   V = mynorm(M,dim);  �ص�dimά��������2����
%   V = mynorm(M);      ����������ģ���൱��V = mynorm(M,1);
if (nargin == 1)
    dim = 1;
elseif (nargin > 2)
    error('only accept inputs.');
end
V = sum(M.^2,dim).^.5;
    