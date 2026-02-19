
def initialize_latex(filename):
    string = r"""\documentclass{article}
    \usepackage{graphicx}
    \usepackage{amsmath}
    \usepackage[english]{babel}
    \usepackage[utf8]{inputenc}
    \title{Notes}
    \date{}
    \begin{document}
    \maketitle
    \end{document}
    """
    with open(filename, "w") as f:
        f.write(string)

def append_latex(filename, new_content):
    """Read the tex file and add the new_content"""
    with open(filename, "r") as f:
        lines = f.readlines()
    
    lines = [line for line in lines if line.strip() != r"\end{document}"]

    lines.append(new_content + "\n")
    lines.append(r"\end{document}")
    
    with open(filename, "w") as f:
        f.writelines(lines)


def get_figure_string(figure_path, figure_width=0.5):
    string = rf"""\begin{{figure}}[!h]
    \centering
    \includegraphics[width={figure_width}\linewidth]{{{figure_path}}}
    \caption{{}}
    \label{{}}
    \end{{figure}}"""
    return string


def export_source(sourceid, source_dataframe, text_notes, figure_paths):
    """
    sourceid : int id
    source_dataframe : pandas DataFrame the dataframe with the relevan columns shown iun the exploring/labeling dashboard
    text_notes : str
    figure_paths : list of str
    """
    

    string = rf"""\section*{{{sourceid}}}""" + "\n"
    string += (source_dataframe.to_latex(escape = True, index=False)+"\n")
    string += (text_notes + "\n")
    for figure_path in figure_paths:
        string += (get_figure_string(figure_path) + "\n")
    
    return string
