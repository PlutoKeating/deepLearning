#!/usr/bin/env python3
"""Convert lab04_report_full.html to PDF with CJK font support."""
from weasyprint import HTML

html_path = "/home/pluto/lab04_dcgan_generation/lab04_report_full.html"
pdf_path = "/home/pluto/lab04_dcgan_generation/202464870331_QinJihe_lab04.pdf"

HTML(filename=html_path).write_pdf(pdf_path)
print(f"PDF written to {pdf_path}")
