"""
First install gdown
After that, you have to zip all file, so you can extract this in Linux Ubuntu environment. How it can be? Go to colab, zip it all.
After that, share the zipped file, so all people can download it.
Next, you will look the link like this ==>> https://drive.google.com/file/d/1kTl9ktvduakKJ5kbUQ79VUwE8VfUzTn6/view?usp=sharing

Copy '1kTl9ktvduakKJ5kbUQ79VUwE8VfUzTn6', set into url like the example below!
"""

import gdown
url = 'https://drive.google.com/uc?id=1kTl9ktvduakKJ5kbUQ79VUwE8VfUzTn6' 
output = 'dataset.zip' 
gdown.download(url, output, quiet=False)
