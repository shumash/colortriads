function res = computeBernstein(n, i, j, k, u, v)
    res = factorial(n) / (factorial(i) * factorial(j) * factorial(k)) .* power(u, i) .* power(v, j) .* power(1.0 - u - v, k);
end

