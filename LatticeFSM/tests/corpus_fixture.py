"""A corpus of every kind the reader knows, for the tests: text, markup, JSON lines, a Word file, an EPUB, gzip,
tar, tar.gz, and zips - deflated and stored members, a ZIP64 one, an archive inside an archive, hidden members,
a picture - built with the standard library."""

import gzip
import io
import json
import os
import tarfile
import zipfile

HTML = """<!doctype html><html><head><title>A page</title><style>p {color: red}</style>
<script>if (a < b) { alert("no") }</script></head><body><!-- not this -->
<h1>Cats &amp; dogs</h1><p>The cat <b>sat</b> on the&nbsp;mat.</p><p>Caf&#233; &#x2014; done&hellip;</p>
<table><tr><td>one</td><td>two</td></tr></table><ul><li>first</li><li>second</li></ul></body></html>"""

DOCX_XML = ('<?xml version="1.0"?><w:document xmlns:w="w"><w:body><w:p><w:r><w:t>The first paragraph</w:t></w:r>'
            '</w:p><w:p><w:r><w:t>and the</w:t><w:tab/><w:t>second.</w:t></w:r></w:p></w:body></w:document>')


def _zip(members, zip64=False, method=zipfile.ZIP_DEFLATED):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", method) as zf:
        for name, data in members:
            if zip64:
                with zf.open(zipfile.ZipInfo(name), "w", force_zip64=True) as f:
                    f.write(data)
            else:
                zf.writestr(name, data)
    return buf.getvalue()


def _tar(members):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tf:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = 0
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _gz(data):
    return gzip.compress(data, mtime=0)


def make(root):
    os.makedirs(os.path.join(root, "more"), exist_ok=True)
    def w(name, data):
        with open(os.path.join(root, name), "wb") as f:
            f.write(data if isinstance(data, bytes) else data.encode())

    w("a.txt", "plain line one\r\nline two\n\n   indented three   \n")
    w("b.html", HTML)
    w("c.docx", _zip([("[Content_Types].xml", "<Types/>"), ("word/document.xml", DOCX_XML)]))
    w("d.epub", _zip([("mimetype", "application/epub+zip"), ("OEBPS/ch1.xhtml", "<html><body><p>Chapter one.</p></body></html>"),
                      ("OEBPS/ch2.html", "<p>Chapter two.</p>")], method=zipfile.ZIP_STORED))
    w("e.jsonl.gz", _gz(("\n".join(json.dumps({"text": t}) for t in ["json one", "json two"]) + "\nnot json\n").encode()))
    long_name = "deep/" + "x" * 120 + ".txt"
    w("f.tar", _tar([("t1.txt", b"tar one\n"), (long_name, b"a long-named member\n"), (".hid.txt", b"hidden\n")]))
    w("g.tgz", _gz(_tar([("in/g.md", b"gzipped tar\n")])))
    w("h.txt.gz", _gz(b"gzipped text\nsecond gz line\n" * 3))
    inner = _gz(_tar([("nested.txt", b"nested in a tar.gz in a zip\n")]))
    w("i.zip", _zip([("x.txt", "zip text\n"), ("sub/y.md", "zip markdown\n"), ("z.html", "<p>zip html</p>"),
                     (".hidden.txt", "no\n"), ("__MACOSX/x.txt", "no\n"), ("img.png", bytes(range(256)) * 4),
                     ("dir/", b""), ("nested.tar.gz", inner), ("inner.zip", _zip([("deeper.txt", "zip in a zip\n")]))]))
    w("j.zip", _zip([("big.txt", ("a ZIP64 member line\n" * 2000).encode())], zip64=True))
    w("k.unknown", b"\x00\x01binary\n")
    w("more/l.srt", "1\n00:00:01,000 --> 00:00:02,000\nA subtitle.\n")
    w("README.md", "not read in a walk\n")
    w("m.odt", _zip([("content.xml", "<office:text><text:p>Open document.</text:p><text:h>A heading</text:h></office:text>")]))
    return root


if __name__ == "__main__":
    import sys
    make(sys.argv[1])
