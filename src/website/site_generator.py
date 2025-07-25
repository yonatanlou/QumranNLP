from jinja2 import Environment, FileSystemLoader
from config import BASE_DIR
from src.website.data_processing import generate_scatter_plots
import os
import subprocess
import sys

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")

fig1, fig2 = generate_scatter_plots()

# Generate interactive clustering analysis by running the scatter plot script
print("Generating interactive clustering analysis...")
try:
    result = subprocess.run([
        sys.executable, "-m", "src.plots_generation.clustering_within_scroll_scatter"
    ], cwd=BASE_DIR, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("Interactive scatter plots generated successfully!")
    else:
        print(f"Warning: Interactive scatter plots generation failed: {result.stderr}")
        
except Exception as e:
    print(f"Error running scatter plot generation: {e}")

env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
template = env.get_template("index.html")

# Render the template with the Plotly figures embedded
html_output = template.render(
    fig1=fig1.to_html(full_html=False, include_plotlyjs=False, div_id="figure1"),
    fig2=fig2.to_html(full_html=False, include_plotlyjs=False, div_id="figure2")
)

# Write the rendered HTML to file
output_path = f"{BASE_DIR}/index.html"
with open(output_path, "w") as f:
    f.write(html_output)

print(f"Enhanced HTML file written to: {output_path}")