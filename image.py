import os
from PIL import Image
import time
start_time = time.time()
#main()
directory = './source'
dir= './destination'

for file_name in os.listdir(directory):
  print("Processing %s" % file_name)
  image = Image.open(os.path.join(directory, file_name))

  x,y = image.size
  new_dimensions = (round(x/2), round(y/2))
  output = image.resize(new_dimensions, Image.ANTIALIAS)
  output_file_name = os.path.join(dir, "small_" + file_name)
  output.save(output_file_name, "JPEG", quality = 95)
  start_time = time.time()
  print("--- %s seconds ---" % (time.time() - start_time))
print("All done")