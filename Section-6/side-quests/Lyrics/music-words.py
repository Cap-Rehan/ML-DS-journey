
# %% [markdown]
# # Imports

# %%
from pathlib import Path
from wordcloud import WordCloud
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# # Variables to set (just change this part)

# %%
md_dir= Path('/Users/mrehanansari/Documents/rehannnn/Lyrics/EdSheeran/Mathematics Collection Tour') # folder containing all markdown files
PNG_NAME= 'sheeranEd.png' # image for mask
bg= 'black' # background color
cmap= 'tab20' # colormap for wordcloud
fc= '#AEB4B9' # color for border (frame)
save_name= "mathematics.png" # name you want for the final exported file 


# %% [markdown]
# # Function to extract content

# %%
def strip_front_matter(text: str) -> str:
    if text.startswith("---"):
        _, _, rest = text.split("---", 2)
        return rest.lstrip()
    return text


# %%
cleaned = []
for file in sorted(md_dir.glob("*.md")):
    text = file.read_text(encoding="utf-8")
    cleaned.append(strip_front_matter(text))

content = "\n\n".join(cleaned)

# %% [markdown]
# # Creating canvas

# %%
icon= Image.open(PNG_NAME)
image_mask= Image.new(mode= "RGB", size= icon.size, color= (255,255,255))
image_mask.paste(icon, box= icon)

# %%
rgb_array= np.array(image_mask)

# %% [markdown]
# # Preparing wordcloud

# %%
word_cloud= WordCloud(width= 3000, height= 4000, background_color= bg,
                    mask= rgb_array, max_words= 500, relative_scaling= 0.5, 
                    min_font_size= 8, colormap= cmap, font_path= 'montserrat.bold.ttf') 
word_cloud.generate(content.upper())

plt.figure(figsize= (12, 15), dpi= 300, facecolor= fc)
plt.imshow(word_cloud, interpolation= 'bilinear')
plt.axis('off')
plt.savefig(
    save_name,
    dpi= 300,
    bbox_inches= "tight",
    facecolor= fc
)

# %%
