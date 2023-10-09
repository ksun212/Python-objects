class StandaloneHTMLBuilder:
    """
    Builds standalone HTML docs.
    """
    # name = 'html'
    # format = 'html'
    # epilog = __('The HTML pages are in %(outdir)s.')

    # copysource = True
    # allow_parallel = True
    # out_suffix = '.html'
    # link_suffix = '.html'  # defaults to matching out_suffix
    # indexer_format: Any = js_index
    # indexer_dumps_unicode = True
    # # create links to original images from images [True/False]
    # html_scaled_image_link = True
    # supported_image_types = ['image/svg+xml', 'image/png',
    #                          'image/gif', 'image/jpeg']
    # supported_remote_images = True
    # supported_data_uri_images = True
    # searchindex_filename = 'searchindex.js'
    # add_permalinks = True
    # allow_sharp_as_current_path = True
    # embedded = False  # for things like HTML help or Qt help: suppresses sidebar
    # search = True  # for things like HTML help and Apple help: suppress search
    # use_index = False
    # download_support = True  # enable download role

    # imgpath: str = None
    # domain_indices: list[DOMAIN_INDEX_TYPE] = []
    globalcontext: dict
    def __init__(self, app: Sphinx, env: BuildEnvironment = None) -> None:
        super().__init__(app, env)

        # CSS files
        self.css_files: list[Stylesheet] = []

        # JS files
        self.script_files: list[JavaScript] = []

        # Cached Publisher for writing doctrees to HTML
        reader = docutils.readers.doctree.Reader(parser_name='restructuredtext')
        pub = Publisher(
            reader=reader,
            parser=reader.parser,
            writer=HTMLWriter(self),
            source_class=DocTreeInput,
            destination=StringOutput(encoding='unicode'),
        )
        if docutils.__version_info__[:2] >= (0, 19):
            pub.get_settings(output_encoding='unicode', traceback=True)
        else:
            op = pub.setup_option_parser(output_encoding='unicode', traceback=True)
            pub.settings = op.get_default_values()
        self._publisher = pub

    def init(self) -> None:
        self.build_info = self.create_build_info()
        # basename of images directory
        self.imagedir = '_images'
        # section numbers for headings in the currently visited document
        self.secnumbers: dict[str, tuple[int, ...]] = {}
        # currently written docname
        self.current_docname: str = None

        self.init_templates()
        self.init_highlighter()
        self.init_css_files()
        self.init_js_files()

        html_file_suffix = self.get_builder_config('file_suffix', 'html')
        if html_file_suffix is not None:
            self.out_suffix = html_file_suffix

        html_link_suffix = self.get_builder_config('link_suffix', 'html')
        if html_link_suffix is not None:
            self.link_suffix = html_link_suffix
        else:
            self.link_suffix = self.out_suffix

        self.use_index = self.get_builder_config('use_index', 'html')

    def create_build_info(self) -> BuildInfo:
        return BuildInfo(self.config, self.tags, ['html'])

    def _get_translations_js(self) -> str:
        candidates = [path.join(dir, self.config.language,
                                'LC_MESSAGES', 'sphinx.js')
                      for dir in self.config.locale_dirs] + \
                     [path.join(package_dir, 'locale', self.config.language,
                                'LC_MESSAGES', 'sphinx.js'),
                      path.join(sys.prefix, 'share/sphinx/locale',
                                self.config.language, 'sphinx.js')]

        for jsfile in candidates:
            if path.isfile(jsfile):
                return jsfile
        return None

    def _get_style_filenames(self) -> Iterator[str]:
        if isinstance(self.config.html_style, str):
            yield self.config.html_style
        elif self.config.html_style is not None:
            yield from self.config.html_style
        elif self.theme:
            stylesheet = self.theme.get_config('theme', 'stylesheet')
            yield from map(str.strip, stylesheet.split(','))
        else:
            yield 'default.css'

    def get_theme_config(self) -> tuple[str, dict]:
        return self.config.html_theme, self.config.html_theme_options

    def init_templates(self) -> None:
        theme_factory = HTMLThemeFactory(self.app)
        themename, themeoptions = self.get_theme_config()
        self.theme = theme_factory.create(themename)
        self.theme_options = themeoptions.copy()
        self.create_template_bridge()
        self.templates.init(self, self.theme)

    def init_highlighter(self) -> None:
        # determine Pygments style and create the highlighter
        if self.config.pygments_style is not None:
            style = self.config.pygments_style
        elif self.theme:
            style = self.theme.get_config('theme', 'pygments_style', 'none')
        else:
            style = 'sphinx'
        self.highlighter = PygmentsBridge('html', style)

        if self.theme:
            dark_style = self.theme.get_config('theme', 'pygments_dark_style', None)
        else:
            dark_style = None

        if dark_style is not None:
            self.dark_highlighter = PygmentsBridge('html', dark_style)
            self.app.add_css_file('pygments_dark.css',
                                  media='(prefers-color-scheme: dark)',
                                  id='pygments_dark_css')
        else:
            self.dark_highlighter = None

    def init_css_files(self) -> None:
        self.css_files = []
        self.add_css_file('pygments.css', priority=200)

        for filename in self._get_style_filenames():
            self.add_css_file(filename, priority=200)

        for filename, attrs in self.app.registry.css_files:
            self.add_css_file(filename, **attrs)

        for filename, attrs in self.get_builder_config('css_files', 'html'):
            attrs.setdefault('priority', 800)  # User's CSSs are loaded after extensions'
            self.add_css_file(filename, **attrs)

    def add_css_file(self, filename: str, **kwargs: Any) -> None:
        if '://' not in filename:
            filename = posixpath.join('_static', filename)

        self.css_files.append(Stylesheet(filename, **kwargs))

    def init_js_files(self) -> None:
        self.script_files = []
        self.add_js_file('documentation_options.js', id="documentation_options",
                         data_url_root='', priority=200)
        self.add_js_file('doctools.js', priority=200)
        self.add_js_file('sphinx_highlight.js', priority=200)

        for filename, attrs in self.app.registry.js_files:
            self.add_js_file(filename, **attrs)

        for filename, attrs in self.get_builder_config('js_files', 'html'):
            attrs.setdefault('priority', 800)  # User's JSs are loaded after extensions'
            self.add_js_file(filename, **attrs)

        if self._get_translations_js():
            self.add_js_file('translations.js')

    def add_js_file(self, filename: str, **kwargs: Any) -> None:
        if filename and '://' not in filename:
            filename = posixpath.join('_static', filename)

        self.script_files.append(JavaScript(filename, **kwargs))

    @property
    def default_translator_class(self) -> type[nodes.NodeVisitor]:  # type: ignore
        if self.config.html4_writer:
            return HTML4Translator  # RemovedInSphinx70Warning
        else:
            return HTML5Translator

    @property
    def math_renderer_name(self) -> str:
        name = self.get_builder_config('math_renderer', 'html')
        if name is not None:
            # use given name
            return name
        else:
            # not given: choose a math_renderer from registered ones as possible
            renderers = list(self.app.registry.html_inline_math_renderers)
            if len(renderers) == 1:
                # only default math_renderer (mathjax) is registered
                return renderers[0]
            elif len(renderers) == 2:
                # default and another math_renderer are registered; prior the another
                renderers.remove('mathjax')
                return renderers[0]
            else:
                # many math_renderers are registered. can't choose automatically!
                return None

    def get_outdated_docs(self) -> Iterator[str]:
        try:
            with open(path.join(self.outdir, '.buildinfo'), encoding="utf-8") as fp:
                buildinfo = BuildInfo.load(fp)

            if self.build_info != buildinfo:
                logger.debug('[build target] did not match: build_info ')
                yield from self.env.found_docs
                return
        except ValueError as exc:
            logger.warning(__('Failed to read build info file: %r'), exc)
        except OSError:
            # ignore errors on reading
            pass

        if self.templates:
            template_mtime = self.templates.newest_template_mtime()
        else:
            template_mtime = 0
        for docname in self.env.found_docs:
            if docname not in self.env.all_docs:
                logger.debug('[build target] did not in env: %r', docname)
                yield docname
                continue
            targetname = self.get_outfilename(docname)
            try:
                targetmtime = path.getmtime(targetname)
            except Exception:
                targetmtime = 0
            try:
                srcmtime = max(path.getmtime(self.env.doc2path(docname)),
                               template_mtime)
                if srcmtime > targetmtime:
                    logger.debug(
                        '[build target] targetname %r(%s), template(%s), docname %r(%s)',
                        targetname,
                        datetime.utcfromtimestamp(targetmtime),
                        datetime.utcfromtimestamp(template_mtime),
                        docname,
                        datetime.utcfromtimestamp(path.getmtime(self.env.doc2path(docname))),
                    )
                    yield docname
            except OSError:
                # source doesn't exist anymore
                pass

    def get_asset_paths(self) -> list[str]:
        return self.config.html_extra_path + self.config.html_static_path

    def render_partial(self, node: Node | None) -> dict[str, str]:
        """Utility: Render a lone doctree node."""
        if node is None:
            return {'fragment': ''}

        doc = new_document('<partial node>')
        doc.append(node)
        self._publisher.set_source(doc)
        self._publisher.publish()
        return self._publisher.writer.parts

    def prepare_writing(self, docnames: set[str]) -> None:
        # create the search indexer
        self.indexer = None
        if self.search:
            from sphinx.search import IndexBuilder
            lang = self.config.html_search_language or self.config.language
            self.indexer = IndexBuilder(self.env, lang,
                                        self.config.html_search_options,
                                        self.config.html_search_scorer)
            self.load_indexer(docnames)

        self.docwriter = HTMLWriter(self)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            # DeprecationWarning: The frontend.OptionParser class will be replaced
            # by a subclass of argparse.ArgumentParser in Docutils 0.21 or later.
            self.docsettings: Any = OptionParser(
                defaults=self.env.settings,
                components=(self.docwriter,),
                read_config_files=True).get_default_values()
        self.docsettings.compact_lists = bool(self.config.html_compact_lists)

        # determine the additional indices to include
        self.domain_indices = []
        # html_domain_indices can be False/True or a list of index names
        indices_config = self.config.html_domain_indices
        if indices_config:
            for domain_name in sorted(self.env.domains):
                domain: Domain = self.env.domains[domain_name]
                for indexcls in domain.indices:
                    indexname = f'{domain.name}-{indexcls.name}'
                    if isinstance(indices_config, list):
                        if indexname not in indices_config:
                            continue
                    content, collapse = indexcls(domain).generate()
                    if content:
                        self.domain_indices.append(
                            (indexname, indexcls, content, collapse))

        # format the "last updated on" string, only once is enough since it
        # typically doesn't include the time of day
        lufmt = self.config.html_last_updated_fmt
        if lufmt is not None:
            self.last_updated = format_date(lufmt or _('%b %d, %Y'),
                                            language=self.config.language)
        else:
            self.last_updated = None

        # If the logo or favicon are urls, keep them as-is, otherwise
        # strip the relative path as the files will be copied into _static.
        logo = self.config.html_logo or ''
        favicon = self.config.html_favicon or ''

        if not isurl(logo):
            logo = path.basename(logo)
        if not isurl(favicon):
            favicon = path.basename(favicon)

        self.relations = self.env.collect_relations()

        rellinks: list[tuple[str, str, str, str]] = []
        if self.use_index:
            rellinks.append(('genindex', _('General Index'), 'I', _('index')))
        for indexname, indexcls, _content, _collapse in self.domain_indices:
            # if it has a short name
            if indexcls.shortname:
                rellinks.append((indexname, indexcls.localname,
                                 '', indexcls.shortname))

        # back up script_files and css_files to allow adding JS/CSS files to a specific page.
        self._script_files = list(self.script_files)
        self._css_files = list(self.css_files)
        styles = list(self._get_style_filenames())

        self.globalcontext = {
            'embedded': self.embedded,
            'project': self.config.project,
            'release': return_codes_re.sub('', self.config.release),
            'version': self.config.version,
            'last_updated': self.last_updated,
            'copyright': self.config.copyright,
            'master_doc': self.config.root_doc,
            'root_doc': self.config.root_doc,
            'use_opensearch': self.config.html_use_opensearch,
            'docstitle': self.config.html_title,
            'shorttitle': self.config.html_short_title,
            'show_copyright': self.config.html_show_copyright,
            'show_search_summary': self.config.html_show_search_summary,
            'show_sphinx': self.config.html_show_sphinx,
            'has_source': self.config.html_copy_source,
            'show_source': self.config.html_show_sourcelink,
            'sourcelink_suffix': self.config.html_sourcelink_suffix,
            'file_suffix': self.out_suffix,
            'link_suffix': self.link_suffix,
            'script_files': self.script_files,
            'language': convert_locale_to_language_tag(self.config.language),
            'css_files': self.css_files,
            'sphinx_version': __display_version__,
            'sphinx_version_tuple': sphinx_version,
            'docutils_version_info': docutils.__version_info__[:5],
            'styles': styles,
            'style': styles[-1],  # xref RemovedInSphinx70Warning
            'rellinks': rellinks,
            'builder': self.name,
            'parents': [],
            'logo_url': logo,
            'favicon_url': favicon,
            'html5_doctype': not self.config.html4_writer,
        }
        if self.theme:
            self.globalcontext.update(
                ('theme_' + key, val) for (key, val) in
                self.theme.get_options(self.theme_options).items())
        self.globalcontext.update(self.config.html_context)
    def handle_page(self, pagename: str, addctx: dict, templatename: str = 'page.html',
                    outfilename: str | None = None, event_arg: Any = None) -> None:
        ctx = self.globalcontext.copy()