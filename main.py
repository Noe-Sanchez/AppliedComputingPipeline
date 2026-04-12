# QoL imports
import termcolor as tc
import matplotlib.pyplot as plt
import pickle
import numpy as np

# Module imports
import src.data_loader as data_loader
import src.regression  as regression
import src.optimizer   as optimizer
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
  best_fit_idx = -1
  best_r2 = 0

  # Data for fits
  variances = [x[1] for x in im_data[:800]]
  altitudes = [flight_data['rel_alt'][i] for i in range(800)] 

  for i in range(4):
    fit = regression.fit_regression(altitudes, variances, i+1)
    if fit[1] > best_r2:
        best_r2 = fit[1]  # fit[1] = r_squared
        best_fit_idx = i
    regression_fits.append(fit)

  print(tc.colored("Regression analysis completed!", bblue))

  # Optimization
  # --- Find the altitude that minimises fitted variance ---
  best_coeffs = regression_fits[best_fit_idx][0]  # np.polyfit order: highest degree first

  # Search window = the altitude range actually observed in the flight.
  alt_min, alt_max = 10.0, 32.5

  # Analytic roots of the derivative (critical points).
  deriv = np.polyder(best_coeffs)
  crit = np.roots(deriv) if len(deriv) > 0 else np.array([])
  crit = crit[np.isreal(crit)].real                    # drop complex roots
  crit = crit[(crit >= alt_min) & (crit <= alt_max)]   # keep in-range

  # Candidates: critical points + the two endpoints.
  candidates = np.concatenate([crit, [alt_min, alt_max]])
  values = np.polyval(best_coeffs, candidates)
  studied_tree_height = 8.7
  optimal_height = float(candidates[np.argmin(values)]) - studied_tree_height

  print(tc.colored(
      f"Optimal relative altitude (min variance): {optimal_height:.2f} m "
      f"(variance = {values.min():.4f})",
      bblue,
  ))

  opt_result = optimizer.run(figures_dir = "report/figures", 
                             processed_dir = "data/processed",
                             clearance=optimal_height)

  print(tc.colored("Plotting results...", bgreen))

  # Create data dict for viz
  data_dict = {
    "flight_data": flight_data,
    "im_data": im_data,
    "regression_fits": regression_fits,
    "opt_tour": opt_result["opt_tour"],
    "global_poses": opt_result["global_poses"],
    "names": opt_result["names"],
    "dist_matrix": opt_result["dist_matrix"],
    "nn_length": opt_result["nn_length"],
    "opt_length": opt_result["opt_length"],
    "history": opt_result["history"],
    "full_trajectory": opt_result["full_trajectory"],
    "history_labels": opt_result["history_labels"],
    "trees": opt_result["trees"],
    "waypoints": opt_result["waypoints"],
    "local_poses": opt_result["local_poses"],
    "tour": opt_result["tour"],
    "figures_dir": opt_result["figures_dir"],
    "clearance": opt_result["clearance"],
    "all_histories": opt_result["all_histories"],
    "segment_lengths": opt_result["segment_lengths"],
    "takeoff_xyz": opt_result["takeoff_xyz"],
  }
    
  viz.generate_report(data_dict)

  print(tc.colored("Report generated successfully!", bblue))

if __name__ == "__main__":
  main()
