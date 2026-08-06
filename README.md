# rikabplotlib

Matplotlib style sheets and helpers for physics plots. 

```bash
pip install rikabplotlib
```

```python
from rikabplotlib import newplot, hist_with_errors, hist_stack, ratio_panel, legend, BLUE, GREEN

fig, main, pull = newplot(ratio = True)

model = hist_stack(main, [continuum, reducible], bins = 40, range = (100, 180), colors = [BLUE, GREEN], labels = ["Continuum", "Reducible"])
data = hist_with_errors(main, measured, bins = 40, range = (100, 180), color = "black", label = "Data")
ratio_panel(pull, data.centers, data.counts, model.counts, data.errors, model.errors, mode = "pull")

main.set_ylabel("Events / 2 GeV")
pull.set_xlabel(r"$m_{\gamma\gamma}$ [GeV]")
legend(main)
```

![Stacked model under the data](https://raw.githubusercontent.com/rikab/rikabplotlib/main/examples/figures/stack.png)

Figures: `newplot`, `set_style`, `add_whitespace`, `stamp`, `badge`, `legend`.


See an example in 
[`examples/rikabplotlib_examples.ipynb`](https://github.com/rikab/rikabplotlib/blob/main/examples/rikabplotlib_examples.ipynb).
