
x=cat(2, [0:0.25:50], [51:10:500]);
y=x;
for i=1:numel(x)
    y(i) = -1 + 2.0 / power(1.0 + exp(1.5*x(i)-6), 1/19.0);
end

N = 5;
p = polyfit(x, y, N);

y2=zeros(size(x));
for i=1:numel(x)
    for j=1:N+1
        y2(i) = y2(i) + power(x(i), N+1-j) * p(j);
    end
end

figure;
plot(x,y);
hold on;
plot(x,y2, 'r');