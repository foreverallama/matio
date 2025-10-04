% Fig file
x = 0:0.1:2*pi;
y = sin(x);

figure;
plot(x, y);
title('Simple Sine Wave');
xlabel('x');
ylabel('sin(x)');

% Save the figure to a .fig file
savefig('test_figure.fig');
close all
