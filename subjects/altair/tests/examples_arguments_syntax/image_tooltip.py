"""
Image Tooltip
-------------
This example shows how to render images in tooltips.
Either URLs or local file paths can be used to reference
the images. To render the image, you must use the special
column name "image" in your data and pass it as a list to
the tooltip encoding.
"""
# category: interactive charts

import altair as alt
import pandas as pd

source = pd.DataFrame.from_records(
    [{'a': 1, 'b': 1, 'image': 'https://altair-viz.github.io/_static/altair-logo-light.png'},
     {'a': 2, 'b': 2, 'image': 'https://avatars.githubusercontent.com/u/11796929?s=200&v=4'}]
)
alt.Chart(source).mark_circle(size=200).encode(
    x='a',
    y='b',
    tooltip=['image']  # Must be a list containing a field called "image"
)
