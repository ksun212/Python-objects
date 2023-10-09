# pypdf vs X

pypdf is a [free] and open source pure-python PDF library capable of
splitting, merging, cropping, and transforming the pages of PDF files.
It can also add custom data, viewing options, and passwords to PDF
files. pypdf can retrieve text and metadata from PDFs as well.

## PyMuPDF and PikePDF

[PyMuPDF] is a Python binding to [MuPDF] and [PikePDF] is the Python
binding to [QPDF].

While both are excellent libraries for various use-cases, using them is
not always possible even when they support the use-case. Both of them
are powered by C libraries which makes installation harder and might
cause security concerns. For MuPDF you might also need to buy a
commercial license.

A core feature of pypdf is that it's pure Python. That means there is
no C dependency. It has been used for over 10 years and for this reason
a lot of support via StackOverflow and examples on the internet.

## pypdf

PyPDF2 was merged back into `pypdf`. The development continues at `pypdf`.

## PyPDF3 and PyPDF4

Developing and maintaining open source software is extremely
time-intensive and in the case of pypdf not paid at all. Having a
continuous support is hard.

pypdf was initially released in 2012 on PyPI and received releases
until 2016. From 2016 to 2022 there was no update - but people were
still using it.

As pypdf is free software, there were attempts to fork it and continue
the development. PyPDF3 was first released in 2018 and still receives
updates. PyPDF4 has only one release from 2018.

I, Martin Thoma, the current maintainer of pypdf, hope that we can
bring the community back to one path of development. Let's see.

  [free]: https://en.wikipedia.org/wiki/Free_software
  [PyMuPDF]: https://pypi.org/project/PyMuPDF/
  [MuPDF]: https://mupdf.com/
  [PikePDF]: https://pypi.org/project/pikepdf/
  [QPDF]: https://github.com/qpdf/qpdf


## pdfminer

[`pdfminer.six`](https://pypi.org/project/pdfminer.six/) is capable of
extracting the [font size](https://stackoverflow.com/a/69962459/562769)
/ font weight (bold-ness). It has no capabilities for writing PDF files.

## pdfrw / pdfminer / pdfplumber

I don't have experience with any of those libraries. Please add a
comparison if you know pypdf and [`pdfrw`](https://pypi.org/project/pdfrw/)!

Please be aware that there is also
[`pdfminer`](https://pypi.org/project/pdfminer/) which is not maintained.
Then there is [`pdfrw2`](https://pypi.org/project/pdfrw2/) which doesn't have
a large community behind it.

And there is also [`pdfplumber`](https://pypi.org/project/pdfplumber/)

## Document Generation

There are (Python) [tools to generate PDF documents](https://github.com/py-pdf/awesome-pdf#generators).
pypdf is not one of them.
