from jinja2 import Environment, FileSystemLoader

from config import BASE_DIR
from src.website.data_processing import generate_scatter_plots

fig1, fig2 = generate_scatter_plots()

env = Environment(loader=FileSystemLoader("templates"))
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
#
# # Write the combined HTML to a file
# output_path = "/Users/yonatanlou/dev/QumranNLP/index1.html"
# with open(output_path, "w") as f:
#     f.write(html_str)
#
# print(f"Enhanced HTML file written to: {output_path}")
