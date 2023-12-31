"""
Pacman Chart
------------
Chart made using ``mark_arc`` and constant values.
This could also be made using 
``alt.Chart(source).mark_arc(color = "gold", theta = (5/8)*np.pi, theta2 = (19/8)*np.pi,radius=100)``.
"""
# category: circular plots

import numpy as np
import altair as alt

alt.Chart().mark_arc(color="gold").encode(
    theta=alt.ThetaDatum((5 / 8) * np.pi).scale(None),
    theta2=alt.Theta2Datum((19 / 8) * np.pi),
    radius=alt.RadiusDatum(100).scale(None),
)
