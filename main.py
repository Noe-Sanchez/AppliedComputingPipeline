# QoL imports
import termcolor as tc

# Module imports
import src.data_loader as data_loader

# Print colors
bgreen = (95, 215, 0)
bblue  = (0, 160, 250)

def main():
  print(tc.colored("Loading flight data...", bgreen))
  flight_data = data_loader.load_data()
  print(tc.colored("Flight data loaded successfully!", bblue))


if __name__ == "__main__":
  main()
