pipeline.py when run processes data from the ESA synchrotron which
produces the iron and nitrogen readings of organisms under microgravity.
pipeline.py processes this data by treatment and control group,
generating comparisons of Fe/N readings by heatmaps. Along with this,
pipeline.py also gives the analyzer the functionality to stamp regions
of interest of varying sizes, showing the mean, standard deviation, and
peak of each in an output file or when the mouse is hovering over it.
Furthermore, the color of the heatmaps will be altered to grayscale to
further identify peaks, which can also be highlighted by boxes. All of
these features can be seen in the Examples folder. The program will
automatically also create graphs and data analyses for each group,
automatically moving to the next one after completion. This can be seen
again in the Examples folder. All these pieces of functionality are
controlled by the keyboard (\"enter\" for stamping, \"up\" and \"down\"
to change box size, p to identify peak values, and \"c\" to switch
colormaps).

The data cannot be shown due to confidentiality reasons, so the code has
to remain somewhat of a black box where it is possible to see what it
does but cannot be run by anyone without special access to the data.
