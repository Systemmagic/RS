%% Hankel-DMD

function [Eigval,Eigvec,bo,X,Y,H] = H_DMD(X,delay)
%构建一个行为d*m，列为n-d+1的hankel矩阵
H = zeros(delay*size(X,1),size(X,2)-delay+1);
%每次循环将X矩阵第k列到n-d+k列（即n-d+1列）的所有行赋值给Hankel矩阵的第m*（k-1）+1到m*k行（举例）
for k=1:delay
H(size(X,1)*(k-1)+1:size(X,1)*k,:) = X(:,k:end-delay+k);
end
% X2 ≈ AX1
X1=H(:,1:end-1); X2=H(:,2:end);
% X2 ≈ AX1,X1 = USV∗，将公式1两边同时右乘VS+，得到X2VS+ ≈ AU 则U'X2VS+ ≈ U'AU = K
[U,S,V]=svd(X1,'econ'); S(S>0)=1./S(S>0); K=U'*X2*V*S;
% 特征值的模表示模式的稳定性：大于1表示增长模式，等于1表示稳定模式，小于1表示衰减模式
% 特征值的相位表示模式的振荡频率。
[y,Eigval]=eig(K);% 求低维矩阵K的特征值和特征向量
Eigvec=U*y;%用投影矩阵U还原高维特征向量
% 系数向量bo表示每个动态模式对初始状态的贡献程度。
bo=pinv(Eigvec)*X1(:,1); X=X1; Y=X2;
end