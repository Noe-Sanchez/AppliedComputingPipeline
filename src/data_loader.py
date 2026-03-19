import pandas as pd
import numpy as np
import pyrtklib as rtk
import pathlib
import math

def load_data(obs_data_path, nav_data_path):
  
  script_dir = pathlib.Path(__file__).parent.parent

  ts = rtk.gtime_t()
  te = rtk.gtime_t()

  popt      = rtk.prcopt_default
  popt.navsys = rtk.SYS_GPS | rtk.SYS_GLO | rtk.SYS_GAL | rtk.SYS_CMP 
  popt.ionoopt = rtk.IONOOPT_BRDC 
  popt.tropopt = rtk.TROPOPT_SAAS
  popt.elmin = 15.0 * math.pi / 180.0

  sopt         = rtk.solopt_t()
  sopt.posf    = rtk.SOLF_LLH
  sopt.outhead = 0
  sopt.timef   = 0
  sopt.timeu   = 3

  fopt = rtk.filopt_t()

  # Make logic and paths cleaner
  infiles = [str(script_dir / obs_data_path), str(script_dir / nav_data_path)]
  outfile = str((script_dir / obs_data_path).parent.parent.parent / "processed" / "output.pos")
  outfile_1dchar = rtk.pyrtklib.Arr1Dchar(outfile)

  pcode = rtk.postpos(ts, te, 0.0, 0.0, popt, sopt, fopt, infiles, 2, outfile_1dchar, "", "")

  # Move assert to driver (?)
  assert pcode == 0, "C1: postpos failed with code {}".format(pcode)

  # Load the output file into a pandas DataFrame
  #with open(outfile, 'r') as f:
  f = open(outfile, 'r')

  lines = f.readlines()
  
  # Ignore first line
  lines = lines[1:]

  # Load only first 4 n-whitespace-separated columns into a DataFrame
  data_lines = [line.split()[1:5] for line in lines]
  
  # Initialize time and height w.r.t the first value of each
  first_time = float(data_lines[0][0])
  first_height = float(data_lines[0][3])

  for line in data_lines:
    line[0] = float(line[0]) - first_time
    line[3] = float(line[3]) - first_height
  
  # Load into DataFrame
  data = pd.DataFrame(data_lines, columns=['time', 'lat', 'lon', 'height'])

  f.close()

  return data
