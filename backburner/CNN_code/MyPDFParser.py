import re

from diffpy.srfit.pdf.pdfparser import PDFParser


class MyPDFParser(PDFParser):

    def parseString(self, patstring):
        PDFParser.parseString(self, patstring)

        # useful regex patterns:
        rx = {"f": r"[-+]?(\d+(\.\d*)?|\d*\.\d+)([eE][-+]?\d+)?"}
        # find where does the data start
        res = re.search(r"^#+ start data\s*(?:#.*\s+)*", patstring, re.M)
        # start_data is position where the first data line starts
        if res:
            start_data = res.end()
        else:
            # find line that starts with a floating point number
            regexp = r"^\s*%(f)s" % rx
            res = re.search(regexp, patstring, re.M)
            if res:
                start_data = res.start()
            else:
                start_data = 0
        header = patstring[:start_data]
        databody = patstring[start_data:].strip()

        # find where the metadata starts
        metadata = ""
        res = re.search(r"^#+\ +metadata\b\n", header, re.M)
        if res:
            metadata = header[res.end() :]
            header = header[: res.start()]

        # parse header
        meta = self._meta

        # rstep
        regexp = r"\b(?:rstep|qsig) *= *(%(f)s)\b" % rx
        res = re.search(regexp, header, re.I)
        if res:
            meta["rstep"] = float(res.groups()[0])
        # rmin
        regexp = r"\b(?:rmin|qsig) *= *(%(f)s)\b" % rx
        res = re.search(regexp, header, re.I)
        if res:
            meta["rmin"] = float(res.groups()[0])
        # rmax
        regexp = r"\b(?:rmax|qsig) *= *(%(f)s)\b" % rx
        res = re.search(regexp, header, re.I)
        if res:
            meta["rmax"] = float(res.groups()[0])
        # adp
        regexp = r"\b(?:adp|qsig) *= *(%(f)s)\b" % rx
        res = re.search(regexp, header, re.I)
        if res:
            meta["adp"] = float(res.groups()[0])
        # self._meta.update({"rmin": rmin, "rmax": rmax})
