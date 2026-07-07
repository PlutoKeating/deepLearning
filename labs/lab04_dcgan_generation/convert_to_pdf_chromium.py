#!/usr/bin/env python3
"""Convert HTML to PDF using Playwright (headless Chromium) for proper CJK rendering."""
import asyncio
from playwright.async_api import async_playwright
from pathlib import Path

async def main():
    html_path = Path("/home/pluto/lab04_dcgan_generation/lab04_report_full.html")
    pdf_path = Path("/home/pluto/lab04_dcgan_generation/202464870331_QinJihe_lab04.pdf")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(f"file://{html_path}", wait_until="networkidle")
        await page.pdf(
            path=str(pdf_path),
            format="A4",
            print_background=True,
            margin={"top": "15mm", "bottom": "15mm", "left": "15mm", "right": "15mm"},
        )
        await browser.close()
    print(f"PDF written to {pdf_path}")
    print(f"Size: {pdf_path.stat().st_size / 1024 / 1024:.1f} MB")

asyncio.run(main())
