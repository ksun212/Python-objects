import docutils
from docutils import nodes
from docutils.io import FileOutput
from docutils.nodes import Element, Node, system_message
from sphinx.builders import Builder
from docutils.frontend import Values
from typing import IO, TYPE_CHECKING, Any, Callable, Generator, cast, Union
from unittest.mock import Mock
import re
class SphinxTranslator(nodes.NodeVisitor):
    """A base class for Sphinx translators.

    This class adds a support for visitor/departure method for super node class
    if visitor/departure method for node class is not found.

    It also provides helper methods for Sphinx translators.

    .. note:: The subclasses of this class might not work with docutils.
              This class is strongly coupled with Sphinx.
    """
    settings: Union[Values, Mock]
    def __init__(self, document: nodes.document, builder: Builder) -> None:
        super().__init__(document)
        self.builder = builder
        self.config = builder.config
        self.settings = document.settings
    def dispatch_visit(self, node: Node) -> None:
        """
        Dispatch node to appropriate visitor method.
        The priority of visitor method is:

        1. ``self.visit_{node_class}()``
        2. ``self.visit_{super_node_class}()``
        3. ``self.unknown_visit()``
        """
        
        for node_class in node.__class__.__mro__:
            method = getattr(self, 'visit_%s' % (node_class.__name__), None)
            if method:
                method(node)
                break
        else:
            super().dispatch_visit(node)

    def dispatch_departure(self, node: Node) -> None:
        """
        Dispatch node to appropriate departure method.
        The priority of departure method is:

        1. ``self.depart_{node_class}()``
        2. ``self.depart_{super_node_class}()``
        3. ``self.unknown_departure()``
        """
        for node_class in node.__class__.__mro__:
            method = getattr(self, 'depart_%s' % (node_class.__name__), None)
            if method:
                method(node)
                break
        else:
            super().dispatch_departure(node)

class TexinfoTranslator(SphinxTranslator):

    ignore_missing_images = False
    builder: TexinfoBuilder

    default_elements = {
        'author': '',
        'body': '',
        'copying': '',
        'date': '',
        'direntry': '',
        'exampleindent': 4,
        'filename': '',
        'paragraphindent': 0,
        'preamble': '',
        'project': '',
        'release': '',
        'title': '',
    }

    def __init__(self, document: nodes.document, builder: TexinfoBuilder) -> None:
        super().__init__(document, builder)
        self.init_settings()

        self.written_ids: set[str] = set()          # node names and anchors in output
        # node names and anchors that should be in output
        self.referenced_ids: set[str] = set()
        self.indices: list[tuple[str, str]] = []    # (node name, content)
        self.short_ids: dict[str, str] = {}         # anchors --> short ids
        self.node_names: dict[str, str] = {}        # node name --> node's name to display
        self.node_menus: dict[str, list[str]] = {}  # node name --> node's menu entries
        self.rellinks: dict[str, list[str]] = {}    # node name --> (next, previous, up)

        self.collect_indices()
        self.collect_node_names()
        self.collect_node_menus()
        self.collect_rellinks()

        self.body: list[str] = []
        self.context: list[str] = []
        self.descs: list[addnodes.desc] = []
        self.previous_section: nodes.section | None = None
        self.section_level = 0
        self.seen_title = False
        self.next_section_ids: set[str] = set()
        self.escape_newlines = 0
        self.escape_hyphens = 0
        self.curfilestack: list[str] = []
        self.footnotestack: list[dict[str, list[collected_footnote | bool]]] = []
        self.in_footnote = 0
        self.in_samp = 0
        self.handled_abbrs: set[str] = set()
        self.colwidths: list[int] = []

    def finish(self) -> None:
        if self.previous_section is None:
            self.add_menu('Top')
        for index in self.indices:
            name, content = index
            pointers = tuple([name] + self.rellinks[name])
            self.body.append('\n@node %s,%s,%s,%s\n' % pointers)
            self.body.append(f'@unnumbered {name}\n\n{content}\n')

        while self.referenced_ids:
            # handle xrefs with missing anchors
            r = self.referenced_ids.pop()
            if r not in self.written_ids:
                self.body.append('@anchor{{{}}}@w{{{}}}\n'.format(r, ' ' * 30))
        self.ensure_eol()
        self.fragment = ''.join(self.body)
        self.elements['body'] = self.fragment
        self.output = TEMPLATE % self.elements

    # -- Helper routines
    def escape_id(self, s: str) -> str:
        """Return an escaped string suitable for node names and anchors."""
        bad_chars = ',:()'
        for bc in bad_chars:
            s = s.replace(bc, ' ')
        if re.search('[^ .]', s):
            # remove DOTs if name contains other characters
            s = s.replace('.', ' ')
        s = ' '.join(s.split()).strip()
        return self.escape(s)
    def init_settings(self) -> None:
        elements = self.elements = self.default_elements.copy()
        elements.update({
            # if empty, the title is set to the first section title
            'title': self.settings.title,
            'author': self.settings.author,
            # if empty, use basename of input file
            'filename': self.settings.texinfo_filename,
            'release': self.escape(self.config.release),
            'project': self.escape(self.config.project),
            'copyright': self.escape(self.config.copyright),
            'date': self.escape(self.config.today or
                                format_date(self.config.today_fmt or _('%b %d, %Y'),
                                            language=self.config.language)),
        })
        # title
        title: str = self.settings.title
        if not title:
            title_node = self.document.next_node(nodes.title)
            title = title_node.astext() if title_node else '<untitled>'
        elements['title'] = self.escape_id(title) or '<untitled>'
        # filename
        if not elements['filename']:
            elements['filename'] = self.document.get('source') or 'untitled'
            if elements['filename'][-4:] in ('.txt', '.rst'):  # type: ignore
                elements['filename'] = elements['filename'][:-4]  # type: ignore
            elements['filename'] += '.info'  # type: ignore
        # direntry
        if self.settings.texinfo_dir_entry:
            entry = self.format_menu_entry(
                self.escape_menu(self.settings.texinfo_dir_entry),
                '(%s)' % elements['filename'],
                self.escape_arg(self.settings.texinfo_dir_description))
            elements['direntry'] = ('@dircategory %s\n'
                                    '@direntry\n'
                                    '%s'
                                    '@end direntry\n') % (
            self.escape_id(self.settings.texinfo_dir_category), entry)
        elements['copying'] = COPYING % elements
        # allow the user to override them all
        elements.update(self.settings.texinfo_elements)

