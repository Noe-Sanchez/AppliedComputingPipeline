import matplotlib.pyplot as plt
import numpy as np

def generate_report(module_data):

  flight_data     = module_data['flight_data']
  regression_fits = module_data['regression_fits']
  im_data         = module_data['im_data']

  rel_alts  = [flight_data['rel_alt'][i] for i in range(800)]
  variances = [im_data[i][1] for i in range(800)]

  # Flight data is a pandas DataFrame, and has rel_alt, Frame, latitude, longitude 
  plt.figure(figsize=(10, 6))
  plt.plot(flight_data['Frame'], flight_data['rel_alt'], label='Relative Altitude', color='blue')
  plt.xlabel('Frame')
  plt.ylabel('Relative Altitude (m)')
  plt.title('Relative Altitude over Time')
  plt.yticks([i for i in range(0, int(float(flight_data['rel_alt'].max())) + 10, 10)])
  plt.savefig('relative_altitude.png')

  for coeffs, R2, RMSE in regression_fits:
    plt.figure(figsize=(10, 6))
    plt.plot(rel_alts, variances, color='red')
    plt.xlabel('Alt')
    plt.ylabel('Variance')
    x = rel_alts
    y = variances
    degree = len(coeffs) - 1 
    plt.title('Variance over Rel_Alt Fit of Degree {}'.format(degree))
    regression_line = np.polyval(coeffs, x)
    plt.plot(x, regression_line, color='blue', label='Fitted Regression Line')
    plt.legend()
    plt.savefig('variance_fitted_degree_{}.png'.format(degree))
    
  residuals = variances - regression_line
  plt.figure(figsize=(10, 6))
  plt.scatter(regression_line, residuals, color='purple')
  plt.axhline(0, color='black', linestyle='--')
  plt.xlabel('Fitted Values')
  plt.ylabel('Residuals')
  plt.title('Residuals vs Fitted Values for Degree {}'.format(degree))
  plt.savefig('residuals_fitted_degree_{}.png'.format(degree))
  
