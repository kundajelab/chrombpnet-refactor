"""The parts of the two report generators that were byte-identical.

``make_html.py`` and ``make_html_bias.py`` are 412 and 325 lines with 143 lines
of identical in-order content between them, including a page shell repeated
three times per file (six copies in total) and a 37-line write-HTML-then-PDF
tail. The literals here were extracted from the sources programmatically rather
than retyped, because the reports are compared byte-for-byte by the regression
harness and a single changed space would show up as a diff.
"""

from __future__ import annotations

import os

#: The page shell, once. The only thing that varied between the six copies was
#: the interpolated title, so it is the sole parameter. Braces are doubled
#: because the originals were f-strings containing literal CSS braces.
_PAGE_SHELL = '\n\t\t<html>\n\t\t\t<head>\n\t\t\t\t<title>{page_title_text}</title>\n\t\t\t</head>\t\n\t\t\t<body>\n\t\t\t\t<h3>{page_title_text}</h3>\n\t\t\t</body>\n\t\t\t<body>\n\t\t\t    <style> \n  \t\t\t\t    table.dataframe, th, td {{font-size:11pt; border:1px solid black; border-collapse:collapse; text-align:center;}}\n \t\t\t\t   th, td {{padding: 5px;}}\n\t\t\t    </style>\n\t\t\t    <style> \n  \t\t\t\t    table.new {{font-size:11pt; border:1px solid black; width:100% ; border-collapse:collapse; text-align:center;}}\n \t\t\t\t   th, td {{padding: 5px;}}\n\t\t\t    </style>\n\t\t\t</body>\n\t\t\t'

#: The closing tag block, byte-identical in both files.
_END_HTML = '\n\t</html>\n\t\t'

#: The WeasyPrint stylesheet. The two copies differed only by a trailing
#: newline. Note "mnargin" in the @page rule: that typo is preserved because it
#: is what the current PDFs are rendered with -- correcting it to "margin"
#: would start applying a page margin and change every report.
_PDF_CSS = '\n\t\t@page {\n                size: 450mm 700mm;\n                mnargin: 0in 0in 0in 0in;\n                }\n        .center {\n                 max-width:25%;\n                 max-height:5%;\n                 display: block;\n                 margin-left: auto;\n                 margin-right: auto;\n                 }\n        .cover {\n    \t\t width: 100%;\n   \t\t\t display: block;\n\t\t\t}\n\t\t\t\n\t\ttable {\n  \t\t\t   font-size: 11pt;\n \t\t\t   font-family: Arial;\n \t\t\t   text-align: center;\n \t\t       width: 100%;\n \t\t       border-collapse: collapse;\n \t\t\t   border: 1px solid silver;\n\t\t\t}\n\t\t\t\n        '


def page_shell(page_title_text: str) -> str:
    """Opening HTML for a report, with the title filled in."""
    return _PAGE_SHELL.format(page_title_text=page_title_text)


def loss_curve(prefix: str, fpx: str, log_name: str) -> str:
    """Plot training and validation loss, and return the image's report path.

    ``log_name`` is the per-model log file: "chrombpnet.log" or "bias.log".
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    loss = pd.read_csv(os.path.join(prefix, "logs/{}{}".format(fpx, log_name)),
                       sep=",", header=0)
    plt.rcParams["figure.figsize"] = 4, 4
    plt.figure()
    plt.plot(loss["epoch"], loss["val_loss"], label="val loss")
    plt.plot(loss["epoch"], loss["loss"], label="train loss")
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.ylabel('Total loss')
    plt.tight_layout()
    plt.savefig(os.path.join(prefix, "evaluation/{}epoch_loss.png".format(fpx)),
                format='png', dpi=300)
    return os.path.join("./", "{}epoch_loss.png".format(fpx))


def write_report(prefix: str, fpx: str, main_html: str, html_prefix: str) -> None:
    """Write the HTML, render the PDF from it, then rewrite the HTML.

    The order is load-bearing. The PDF is rendered from an HTML file whose
    image references are still relative ("./"), so WeasyPrint resolves them
    against the evaluation directory and embeds them. Only afterwards is the
    HTML rewritten with the user's --html-prefix, which is for viewing the page
    somewhere else. Rewriting first would leave the PDF without its images.
    """
    from weasyprint import CSS, HTML

    main_html = main_html + _END_HTML
    html_path = os.path.join(
        prefix, "evaluation/{}overall_report.html".format(fpx))
    with open(html_path, 'w') as handle:
        handle.write(main_html)

    HTML(html_path).write_pdf(
        os.path.join(prefix, "evaluation/{}overall_report.pdf".format(fpx)),
        stylesheets=[CSS(string=_PDF_CSS)])

    with open(html_path, 'w') as handle:
        handle.write(main_html.replace("./", html_prefix))
