import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error
from scipy.optimize import curve_fit



data = np.loadtxt("train_data.csv", delimiter=",", skiprows=1)

x = data[:, 0]
y = data[:, 1]


# [ 0.46961745  6.56747706 -9.60341052  2.00397062 -4.44482766  1.75822761  12.33316284  6.39951974 ]
# [ 0.44380184  6.375698   -9.46545533  2.00254918 -4.40415329  1.49184138  8.57221663   7.45124644 ]
# [ 0.41003657  9.03357305 -9.61408771  2.008737   -4.48959462  2.10070974  14.73403763  5.17656052 ]



def sinusoidal(x, A, B, C, D):
   
    return A * x + B + (C) * np.sin(x / 2 -5) + D * np.cos(x /15 -5)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


best_mse = float('inf')
#best_params = [-9.6, 2, -4.4, 1.5, 12.3, 6]
best_params = [0.41003657, 9.03357302, 10.9, 2.26645992]
print(best_params)
iteration = 0
max_iterations = 100



while iteration < max_iterations:
    try:
        params, _ = curve_fit(sinusoidal, x_train, y_train, p0=best_params)
        y_pred = sinusoidal(x_test, *params)
        mse = np.mean((y_test - y_pred)**2)

        if mse < best_mse:
            best_mse = mse
            best_params = params
    
    except RuntimeError:
        pass
    
    iteration += 1


print(f'Mean Squared Error : {mse:.8f}')
print(params)


plt.scatter(x, y, label='Data', color='blue')
plt.plot(np.sort(x), sinusoidal(np.sort(x), *best_params), label='Fitted Curve', color='green')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Fitted Model to Data')
plt.show()

