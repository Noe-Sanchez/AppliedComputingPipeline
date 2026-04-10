# QoL imports
import termcolor as tc
import matplotlib.pyplot as plt
import pickle
import numpy as np

# Module imports
import src.data_loader as data_loader
import src.regression  as regression
import src.viz         as viz

# Print colors
bgreen = (95, 215, 0)
bblue  = (0, 160, 250)
bred   = (255, 0, 0)

def main():
  print(tc.colored("Loading flight data...", bgreen))

  video_path = "data/raw/flight.mp4"
  srt_path   = "data/raw/flight.srt"

  flight_data = data_loader.load_data(video_path, srt_path, True)

  if flight_data is None:
    print(tc.colored("Failed to load flight data.", (255, 0, 0)))
    return
  
  print(tc.colored("Flight data loaded successfully!", bblue))
  print(tc.colored("Performing regression analysis...", bgreen))

  # Check if im_data.pkl exists, if not create it
  try:
    with open('im_data.pkl', 'rb') as f:
      im_data = pickle.load(f)
    print(tc.colored("Loaded im_data of C5 from pickle.", bblue))
  except FileNotFoundError:
    print(tc.colored("im_data.pkl not found. Mocking im_data...", bgreen))
    im_data = regression.mock_data()
    with open('im_data.pkl', 'wb') as f:
      pickle.dump(im_data, f)
    print(tc.colored("C5 im_data mocked and saved to pickle.", bblue))

  # Generate fits
  regression_fits = []

  # Data for fits
  variances = [x[1] for x in im_data[:800]]
  altitudes = [flight_data['rel_alt'][i] for i in range(800)] 

  for i in range(4):
    fit = regression.fit_regression(altitudes, variances, i+1)
    regression_fits.append(fit)

  print(tc.colored("Regression analysis completed!", bblue))
  print(tc.colored("Plotting results...", bgreen))

  # Create data dict for viz
  data_dict = {
    "flight_data": flight_data,
    "im_data": im_data,
    "regression_fits": regression_fits
  }
    
  viz.generate_report(data_dict)

  print(tc.colored("Report generated successfully!", bblue))

if __name__ == "__main__":
  main()
