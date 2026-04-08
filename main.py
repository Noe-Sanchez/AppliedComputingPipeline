# QoL imports
import termcolor as tc

# Module imports
import src.data_loader as data_loader

# Print colors
bgreen = (95, 215, 0)
bblue  = (0, 160, 250)
bred   = (255, 0, 0)

def main():
  print(tc.colored("Loading flight data...", bgreen))

  obs_data = ("./data/raw/flight_1/DJI_202602161725_004_PPKOBS.obs")
  nav_data = ("./data/raw/flight_1/DJI_202602161725_004_PPKNAV.nav")
  flight_data = data_loader.load_data(obs_data, nav_data)

  if flight_data is None:
    print(tc.colored("Failed to load flight data.", (255, 0, 0)))
    return
  
  print(tc.colored("Flight data loaded successfully!", bblue))

  print(flight_data.head())

if __name__ == "__main__":
  main()
