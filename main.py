# QoL imports
import termcolor as tc
import matplotlib.pyplot as plt

# Module imports
import src.data_loader as data_loader

# Print colors
bgreen = (95, 215, 0)
bblue  = (0, 160, 250)
bred   = (255, 0, 0)

def main():
  print(tc.colored("Loading flight data...", bgreen))

  video_path = "data/raw/flight.mp4"
  srt_path   = "data/raw/flight.srt"

  flight_data = data_loader.load_data(video_path, srt_path)
  # For dry run
  #flight_data = data_loader.load_data(video_path, srt_path, True)

  if flight_data is None:
    print(tc.colored("Failed to load flight data.", (255, 0, 0)))
    return
  
  print(tc.colored("Flight data loaded successfully!", bblue))

  print(flight_data)

  # Flight data is a pandas DataFrame, and has rel_alt, Frame, latitude, longitude 
  plt.figure(figsize=(10, 6))
  plt.plot(flight_data['Frame'], flight_data['rel_alt'], label='Relative Altitude', color='blue')
  plt.xlabel('Frame')
  plt.ylabel('Relative Altitude (m)')
  plt.title('Relative Altitude over Time')
  plt.yticks([i for i in range(0, int(float(flight_data['rel_alt'].max())) + 10, 10)])
  plt.savefig('relative_altitude.png')

if __name__ == "__main__":
  main()
