function [ w, Aeq, beq ] = blobLinOptimize(S, T)
%BLOBLINOPTIMIZE Optimizes for weights of the Source
%blobs to reproduce Target.
%S is Nx3 and T is Mx3.

s = size(S)
N = s(1);
ss = size(T)
M = ss(1);

Aeq_constraint = zeros(M, M * N);
Aeq = zeros(M * 3, M * N);
size(Aeq);
Acomp = zeros(3, N);
Acomp(1, :) = S(:, 1)';
Acomp(2, :) = S(:, 2)';
Acomp(3, :) = S(:, 3)';
for i = 1:M
    from_row = (i - 1) * 3 + 1;
    to_row = from_row + 3 - 1;
    from_col = (i - 1) * N + 1;
    to_col = from_col + N - 1;
    Aeq(from_row:to_row, from_col:to_col) = Acomp;
    Aeq_constraint(i, (N*(i-1) + 1):(N*(i-1) + N)) = 1;
end
beq = reshape(T', [M * 3, 1]);

% Reformulate equality as inequality
eps = 0.1;
A = [Aeq; -Aeq];
b = [ beq + eps; -beq + eps]; 


lb = zeros(M * N, 1);
ub = ones(M * N, 1);

f = ones(M * N, 1);
size(f)

options = optimoptions('linprog','Algorithm','dual-simplex');
w = linprog(f, A, b, Aeq_constraint, ones(M, 1), lb, ub, options);
%[], [], lb, ub);
end

