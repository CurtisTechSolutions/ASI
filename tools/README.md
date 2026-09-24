# tools

Scripts that gather material for the experiments in this repository rather than
belonging to any one of them. Standard library only, Python 3.10 or newer: each
runs straight out of a checkout with nothing to install.

| file | what it is |
|---|---|
| `gutenberg_download.py` | plain-text books from [Project Gutenberg](https://www.gutenberg.org/), chosen by language, author, title, subject, bookshelf, Library of Congress class, release date or the authors' lifetimes |
| `wikipedia_download.py` | all of Wikipedia from the official dumps (resumable, checksummed) turned into plain text, or chosen articles through the API |
| `common.py` | what the two share: a polite HTTP client, word matching, safe file names, atomic writes, manifests |
| `tests/` | the tests - offline, a couple of seconds |

Both write a manifest beside what they download - what each file is, where it
came from, a SHA-256 - so a corpus built with them can be checked and rebuilt.

## gutenberg_download.py

```bash
python3 tools/gutenberg_download.py --facet bookshelf -l en             # what is there to filter on?
python3 tools/gutenberg_download.py -l en -b "science fiction" --list   # what would I get?
python3 tools/gutenberg_download.py -l en -b "science fiction" -n 20    # get twenty of them
python3 tools/gutenberg_download.py -a "jane austen" --strip -o austen  # her books, without the licence
python3 tools/gutenberg_download.py -l de --author-alive-to 1900 --sort random --seed 7 -n 100
```

Choosing books asks nothing of the site: the filters run over Project
Gutenberg's own catalog, `pg_catalog.csv`, which is downloaded once and cached
for a week (`--cache-dir`, default `~/.cache/gutenberg_download`; `--refresh`
to fetch it again, `--catalog` to use a copy of your own).

### Filters

| option | keeps a book when | example |
|---|---|---|
| `-l`, `--language CODE` | it is in that language (any of its languages) | `-l en`, `-l en,fr` |
| `-a`, `--author TEXT` | an author, translator, editor or illustrator matches | `-a "mark twain"` |
| `-t`, `--title TEXT` | the title matches | `-t sherlock` |
| `-s`, `--subject TEXT` | a Library of Congress subject heading matches | `-s detective` |
| `-b`, `--bookshelf TEXT` | a Project Gutenberg bookshelf matches | `-b "science fiction"` |
| `--locc CLASS` | a Library of Congress class starts with CLASS | `--locc PR` (English literature), `--locc Q` (science) |
| `--ids LIST` | its number is listed | `--ids 84,1342,1-100,70000-` |
| `--released-from DATE`, `--released-to DATE` | Project Gutenberg released it then, inclusive (`YYYY`, `YYYY-MM` or `YYYY-MM-DD`) | `--released-from 2024` |
| `--author-alive-from YEAR`, `--author-alive-to YEAR` | one of its authors - not its translators - was alive then; negative years are BCE | `--author-alive-from 1800 --author-alive-to 1899` |
| `--type TYPE` | the catalog type is TYPE: `Text` unless you say otherwise, `any` for all | `--type any` |
| `--exclude-author`, `--exclude-title`, `--exclude-subject`, `--exclude-bookshelf` | nothing of that kind matches | `--exclude-subject poetry` |

Text is matched by words, ignoring case and accents: every word you give must
begin a word of the field, in any order. So `-a "austen jane"` finds "Austen,
Jane, 1775-1817", `-a bronte` finds "Brontë", and `-s detect` finds "Detective
and mystery stories"; `--regex` makes the text filters regular expressions
instead. Repeat an option for alternatives (`-l en -l fr`); different options
must all hold. `--facet FIELD` counts the values of `language`, `author`,
`subject`, `bookshelf`, `locc` or `type` among the matching books, which is how
to find out what to type.

The catalog carries no download counts, so "the most popular" is not something
these filters can select.

### What you get

`--sort id|title|author|released|random` picks the order (`--seed` makes a
random sample repeatable) and `-n` stops once that many books are on disk.
Each book is saved as `<id>-<title>.txt` - `--name-format` takes `{id}`,
`{title}`, `{author}` and `{language}`, `/` for folders, and must keep `{id}` -
as UTF-8 with `\n` line endings, whatever it was published in. `--strip` cuts
off the Project Gutenberg header and licence. `manifest.jsonl` holds a line per
book: its catalog entry, the URL, the encoding it arrived in, whether it was
stripped, and the SHA-256 of the file written.

Running the same command again fetches nothing new: books already on disk are
skipped and count towards `-n`, books that failed are tried again, and a book
with no plain-text file is passed over for the next one. The text comes from the
first of the generated UTF-8 file (`cache/epub/<id>/pg<id>.txt`) and the files
volunteers uploaded (`<id>-0.txt`, `<id>.txt`, `<id>-8.txt`).

### Going gently

Project Gutenberg's [robot policy](https://www.gutenberg.org/policy/robot_access.html)
asks automated clients to go slowly and to use a mirror for anything big.
Requests go one at a time, `--delay` 2 seconds apart (their own example is
`wget -w 2`); a server error is retried with growing pauses, and a refusal (403)
stops the run at once rather than knock again. For thousands of books use
`--mirror` (see [the list](https://www.gutenberg.org/MIRRORS.ALL)); for the whole
collection, [mirror it](https://www.gutenberg.org/help/mirroring.html) - the tool
will not download the whole catalog without a filter or `-n`.

## wikipedia_download.py

```bash
python3 tools/wikipedia_download.py dump --lang simple --list    # the newest Simple English dump, without downloading
python3 tools/wikipedia_download.py dump --lang simple           # download it
python3 tools/wikipedia_download.py extract wikipedia_dumps/simplewiki-*-pages-articles.xml.bz2
python3 tools/wikipedia_download.py fetch -c "Quantum mechanics" --depth 1 -n 300
python3 tools/wikipedia_download.py fetch -s "black hole" -n 20 --format jsonl
```

All of Wikipedia, as text, is two commands: `dump`, then `extract`. It is not a
crawl. Wikimedia publishes the whole database twice a month at
[dumps.wikimedia.org](https://dumps.wikimedia.org/) and
[asks](https://en.wikipedia.org/wiki/Wikipedia:Database_download) that it be used
instead of crawling the site - and the dump is also the only way to get all of it
in hours rather than months. `fetch` is the other road, for a few hundred or a few
thousand chosen articles, through the API.

### dump

Finds the newest *finished* dump - a dump run takes days, so the newest date is
often still in progress - prints its files and their size, checks the disk has
room, and downloads them one after another into `wikipedia_dumps/`. Interrupt it
and run it again: it resumes where it stopped, and each finished file is checked
against the SHA-1 Wikimedia publishes. A file that does not match is deleted, not
kept.

| option | |
|---|---|
| `--lang CODE`, `--wiki NAME` | which wiki: `en`, `de`, `simple`, ... or any dump name, such as `enwiktionary` |
| `--kind` | `articles` (the default: the current text of every article), `multistream` (the same, in seekable blocks with an index), `meta-current` (talk and user pages too), `history` (every revision - terabytes), `abstracts`, `titles` |
| `--date YYYYMMDD` | a particular dump rather than the newest finished one |
| `--parts` | the numbered parts big wikis are split into, rather than one whole file |
| `--list` | show the files and sizes; download nothing |
| `--mirror URL` | a [mirror](https://dumps.wikimedia.org/mirrors.html) of dumps.wikimedia.org |

English Wikipedia's `articles` dump is a single file of over 20 GB, bzip2-compressed.

### extract

Streams a dump - `.xml.bz2`, `.xml.gz` or `.xml`, whole or all its parts - into
plain text without ever holding it in memory, converting on `-p` processes (all
cores but one, up to eight, by default).

The output is JSON lines in shards of about 100 MB (`wiki_00000.jsonl`, ...;
`--shard-mb`, `--gzip`), one article per line:
`{"id", "title", "url", "revision", "timestamp", "categories", "text"}`. With
`--format txt` it is a file per article, the title then the text, a thousand to a
folder. `manifest.json` records the dumps, the options, and how many pages were
kept and why the rest were skipped.

It keeps articles (namespace 0; `--namespace` for others) and skips redirects
and disambiguation pages (`--keep-disambiguation`). `-t` and `--exclude-title`,
`-c` and `--exclude-category` (the categories a page puts itself in) match words
the way the Gutenberg filters do; `--min-chars` drops stubs; `-n` stops early.

Templates are removed, not expanded: expanding them needs every template and
Lua module on the wiki, which is to say MediaWiki itself. A handful that carry
words in running text (`{{convert}}`, `{{lang}}`, `{{nowrap}}`, `{{circa}}`, ...)
are rendered instead. Tables, references, files, categories and comments go,
links become their labels, headings become lines of their own, and References,
External links, See also and their like are dropped (`--keep-sections`). The
result is the prose of the article, but not a copy of the rendered page: where a
sentence took words from any other template, there is a gap.

### fetch

| option | which articles |
|---|---|
| `-t TITLE`, `--titles-file FILE` | by title (redirects followed) |
| `-c NAME`, `--depth N` | those in a category, and N levels of its subcategories |
| `-s QUERY` | those a search finds (the API stops at 10,000) |
| `--random N` | N at random |

Sources combine, and an article found twice is saved once. The text is
Wikipedia's own plain-text rendering (TextExtracts), which does expand
templates. Requests go one at a time, `--delay` 0.5 seconds apart, one article
each (`--intro` takes only each lead section, 20 to a request), with `maxlag=5`
so that busy servers can answer "later" - and are then waited for. Articles are
saved as `<pageid>-<title>.txt` in `wikipedia_articles/`, or as lines of
`articles.jsonl` with `--format jsonl`, with a `manifest.jsonl`; `-n`,
`--min-chars` and `--keep-sections` work as above, and running again skips what
is already there. `--lang`, `--site` and `--api` reach other Wikipedias, other
wikis, and any MediaWiki.

Wikimedia's [User-Agent policy](https://meta.wikimedia.org/wiki/User-Agent_policy)
asks automated clients to say how to reach whoever runs them: pass
`--contact you@example.org`, or set `WIKIMEDIA_CONTACT`.

## Tests

```bash
python3 -m unittest discover -s tools/tests    # from the repository root
```

The sites are stood in for: the Gutenberg catalog and books, dumps.wikimedia.org's
listings and `dumpstatus.json`, and a small MediaWiki API. Two test classes go
through a real HTTP server on 127.0.0.1, so urllib's own sockets, gzip and
ranges are exercised as well.
