from pylab import * 
import matplotlib.colors as colors


x = np.array([0.0, 1.0, 2.0, 3.0,  4.0,  5.0])
y = np.array([0.0, 1.5, 4.5, 6.1, 8.4, 10.0])
fit = np.polyfit(x, y, 1)
fit_fn = poly1d(fit)

print fit
print fit_fn(0), fit_fn(1)

plot(x,y, 'yo',  x, fit_fn(x), '--k')
xlim(0, 5)
ylim(0, 12)
show()